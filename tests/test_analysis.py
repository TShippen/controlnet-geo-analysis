"""Tests for the analysis service: validation, caching, and processor dispatch."""

import dataclasses
import io
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from conftest import SAMPLE_MEASUREMENT, sampled_test_spec, write_test_image
from controlnet_mcp.analysis import AnalysisService, ResolutionError
from controlnet_mcp.cache import AnalysisCache
from controlnet_mcp.checkpoints import ZOE_CHECKPOINT, CheckpointSpec
from controlnet_mcp.comparison import ComparisonError
from controlnet_mcp.config import MeasurementSetting, Settings
from controlnet_mcp.images import ReferenceImageError, image_to_png_bytes, png_measurement
from controlnet_mcp.measurements import Measurement
from controlnet_mcp.model_manager import MissingCheckpointError, ModelManager
from controlnet_mcp.processors import (
    PROCESSORS,
    AnalysisOptions,
    AnalysisOutput,
    OptionError,
    ProcessorSpec,
    UnknownAnalysisError,
)
from controlnet_mcp.regions import FULL_IMAGE, CropError, CropRegion
from controlnet_mcp.sampling import SamplingError
from controlnet_mcp.segmentation import PromptError, RegionPrompt

DISTINCT_FORMS = Measurement(brief="B", full="F")


@dataclass
class RunCounter:
    """Records how many times a fake processor ran."""

    calls: int = 0
    resolutions: list[int] = field(default_factory=list)
    prompts: list[RegionPrompt | None] = field(default_factory=list)
    regions: list[CropRegion] = field(default_factory=list)
    image_sizes: list[tuple[int, int]] = field(default_factory=list)
    options: list[AnalysisOptions] = field(default_factory=list)


def make_fake_spec(
    kind: str,
    counter: RunCounter,
    checkpoints: tuple[CheckpointSpec, ...] = (),
    accepts_prompt: bool = False,
    measurement: Measurement = SAMPLE_MEASUREMENT,
    accepts_line_length: bool = False,
) -> ProcessorSpec:
    """A processor that renders a plain image and reports ``measurement``.

    The measurement defaults to a non-empty one because every real processor
    measures its output. A result without one is still written to the cache,
    but it is never served from it.
    """

    def build(model_dir: Path, device: torch.device) -> object:
        return object()

    def run(
        detector: object,
        image: Image.Image,
        resolution: int,
        prompt: RegionPrompt | None,
        region: CropRegion,
        options: AnalysisOptions,
    ) -> AnalysisOutput:
        counter.calls += 1
        counter.resolutions.append(resolution)
        counter.prompts.append(prompt)
        counter.regions.append(region)
        counter.image_sizes.append(image.size)
        counter.options.append(options)
        rendered = Image.new("RGB", (resolution, resolution // 2), (0, 0, 255))
        return AnalysisOutput(rendered, measurement)

    return ProcessorSpec(
        kind=kind,
        description=f"Fake {kind} analysis.",
        use_when=f"Use fake {kind} for tests.",
        checkpoints=checkpoints,
        build=build,
        run=run,
        accepts_prompt=accepts_prompt,
        accepts_line_length=accepts_line_length,
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    (tmp_path / "references").mkdir()
    (tmp_path / "models").mkdir()
    return Settings(
        reference_image_dir=tmp_path / "references",
        model_dir=tmp_path / "models",
        output_dir=tmp_path / "outputs",
        default_detect_resolution=128,
        max_loaded_models=1,
        device="cpu",
    )


@pytest.fixture
def service(settings: Settings) -> AnalysisService:
    manager = ModelManager(settings.model_dir, torch.device("cpu"), max_loaded=1)
    return AnalysisService(settings, manager, AnalysisCache(settings.output_dir))


def service_measuring(settings: Settings, mode: MeasurementSetting) -> AnalysisService:
    """A service over the same directories that emits measurements at ``mode``."""
    tuned = settings.model_copy(update={"result_measurements": mode})
    manager = ModelManager(tuned.model_dir, torch.device("cpu"), max_loaded=1)
    return AnalysisService(tuned, manager, AnalysisCache(tuned.output_dir))


@pytest.fixture
def reference(settings: Settings) -> str:
    write_test_image(settings.reference_image_dir / "box.png", size=(128, 64))
    return "box.png"


def test_analyze_canny_returns_png_with_dimensions(
    service: AnalysisService, reference: str
) -> None:
    result = service.analyze(reference, "canny", 64)

    assert result.png[:8] == b"\x89PNG\r\n\x1a\n"
    assert (result.width, result.height) == (128, 64)
    assert result.from_cache is False
    assert result.kind == "canny"


def test_analyze_second_call_hits_cache(
    service: AnalysisService, reference: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = RunCounter()
    monkeypatch.setitem(PROCESSORS, "fake", make_fake_spec("fake", counter))

    first = service.analyze(reference, "fake", 64)
    second = service.analyze(reference, "fake", 64)

    assert counter.calls == 1
    assert first.from_cache is False
    assert second.from_cache is True
    assert second.png == first.png
    assert "cache" in second.description


def test_analyze_rejects_unknown_kind(service: AnalysisService, reference: str) -> None:
    with pytest.raises(UnknownAnalysisError):
        service.analyze(reference, "pose", 64)


def test_analyze_rejects_resolution_out_of_range(service: AnalysisService, reference: str) -> None:
    with pytest.raises(ResolutionError):
        service.analyze(reference, "canny", 32)


def test_analyze_uses_default_resolution(
    service: AnalysisService, reference: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = RunCounter()
    monkeypatch.setitem(PROCESSORS, "fake", make_fake_spec("fake", counter))

    result = service.analyze(reference, "fake", None)

    assert counter.resolutions == [128]
    assert result.resolution == 128
    assert list(service.settings.output_dir.rglob("fake-v1-128.png"))


def test_version_bump_retires_cached_result(
    service: AnalysisService, reference: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = RunCounter()
    spec = make_fake_spec("fake", counter)
    monkeypatch.setitem(PROCESSORS, "fake", spec)
    service.analyze(reference, "fake", 64)

    monkeypatch.setitem(PROCESSORS, "fake", dataclasses.replace(spec, version="2"))
    bumped = service.analyze(reference, "fake", 64)

    assert counter.calls == 2
    assert bumped.from_cache is False


def test_analyze_missing_checkpoint_propagates(
    service: AnalysisService, reference: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = RunCounter()
    monkeypatch.setitem(PROCESSORS, "fake", make_fake_spec("fake", counter, (ZOE_CHECKPOINT,)))

    with pytest.raises(MissingCheckpointError):
        service.analyze(reference, "fake", 64)
    assert counter.calls == 0


def test_cached_result_reports_png_dimensions(service: AnalysisService, reference: str) -> None:
    service.analyze(reference, "canny", 64)

    cached = service.analyze(reference, "canny", 64)

    assert Image.open(io.BytesIO(cached.png)).size == (cached.width, cached.height) == (128, 64)


def test_segments_requires_prompt(
    service: AnalysisService, reference: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = RunCounter()
    monkeypatch.setitem(PROCESSORS, "fake", make_fake_spec("fake", counter, accepts_prompt=True))

    with pytest.raises(PromptError, match="box"):
        service.analyze(reference, "fake", 64, None)
    assert counter.calls == 0


def test_prompt_rejected_for_whole_image_analysis(service: AnalysisService, reference: str) -> None:
    with pytest.raises(PromptError, match="whole image"):
        service.analyze(reference, "canny", 64, RegionPrompt.from_lists(None, [0.5, 0.5]))


def test_different_prompts_cache_separately(
    service: AnalysisService, reference: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = RunCounter()
    monkeypatch.setitem(PROCESSORS, "fake", make_fake_spec("fake", counter, accepts_prompt=True))
    first = RegionPrompt.from_lists([0.1, 0.1, 0.5, 0.5], None)
    second = RegionPrompt.from_lists([0.5, 0.5, 0.9, 0.9], None)

    service.analyze(reference, "fake", 64, first)
    service.analyze(reference, "fake", 64, second)
    repeat = service.analyze(reference, "fake", 64, first)

    assert counter.calls == 2
    assert repeat.from_cache is True


def test_crop_passes_cropped_image_and_snapped_region(
    service: AnalysisService, reference: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reference is 128x64, so its right half is a 64x64 crop."""
    counter = RunCounter()
    monkeypatch.setitem(PROCESSORS, "fake", make_fake_spec("fake", counter))

    result = service.analyze(
        reference, "fake", 64, crop=CropRegion.from_list([0.5, 0.0, 1.0, 1.0])
    )

    assert counter.image_sizes == [(64, 64)]
    assert counter.regions == [CropRegion(0.5, 0.0, 1.0, 1.0)]
    assert result.crop == CropRegion(0.5, 0.0, 1.0, 1.0)


def test_uncropped_run_receives_full_image(
    service: AnalysisService, reference: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = RunCounter()
    monkeypatch.setitem(PROCESSORS, "fake", make_fake_spec("fake", counter))

    result = service.analyze(reference, "fake", 64)

    assert counter.regions == [FULL_IMAGE]
    assert result.crop is None


def test_crop_rejected_for_prompted_analysis(
    service: AnalysisService, reference: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = RunCounter()
    monkeypatch.setitem(PROCESSORS, "fake", make_fake_spec("fake", counter, accepts_prompt=True))

    with pytest.raises(CropError, match="box or point"):
        service.analyze(
            reference,
            "fake",
            64,
            RegionPrompt.from_lists(None, [0.5, 0.5]),
            CropRegion.from_list([0.0, 0.0, 0.5, 1.0]),
        )
    assert counter.calls == 0


def test_different_crops_cache_separately(
    service: AnalysisService, reference: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = RunCounter()
    monkeypatch.setitem(PROCESSORS, "fake", make_fake_spec("fake", counter))
    left = CropRegion.from_list([0.0, 0.0, 0.5, 1.0])
    right = CropRegion.from_list([0.5, 0.0, 1.0, 1.0])

    service.analyze(reference, "fake", 64, crop=left)
    service.analyze(reference, "fake", 64, crop=right)
    repeat = service.analyze(reference, "fake", 64, crop=left)

    assert counter.calls == 2
    assert repeat.from_cache is True
    assert repeat.crop == left


def test_line_length_rejected_for_other_analyses(
    service: AnalysisService, reference: str
) -> None:
    with pytest.raises(OptionError, match="applies only to lines"):
        service.analyze(reference, "canny", 64, options=AnalysisOptions(line_length="long"))


def test_line_length_reaches_the_run_and_caches_separately(
    service: AnalysisService, reference: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = RunCounter()
    monkeypatch.setitem(
        PROCESSORS, "fake", make_fake_spec("fake", counter, accepts_line_length=True)
    )
    long_only = AnalysisOptions(line_length="long")

    service.analyze(reference, "fake", 64)
    service.analyze(reference, "fake", 64, options=long_only)
    repeat = service.analyze(reference, "fake", 64, options=long_only)

    assert counter.calls == 2
    assert counter.options[1] == long_only
    assert repeat.from_cache is True


def test_line_length_all_shares_the_default_cache_entry(
    service: AnalysisService, reference: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = RunCounter()
    monkeypatch.setitem(
        PROCESSORS, "fake", make_fake_spec("fake", counter, accepts_line_length=True)
    )

    service.analyze(reference, "fake", 64)
    repeat = service.analyze(reference, "fake", 64, options=AnalysisOptions(line_length="all"))

    assert counter.calls == 1
    assert repeat.from_cache is True


def test_measurement_survives_cache(
    service: AnalysisService, reference: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = RunCounter()
    measured = make_fake_spec("fake", counter, accepts_prompt=True, measurement=SAMPLE_MEASUREMENT)
    monkeypatch.setitem(PROCESSORS, "fake", measured)
    prompt = RegionPrompt.from_lists(None, [0.3, 0.3])

    fresh = service.analyze(reference, "fake", 64, prompt)
    cached = service.analyze(reference, "fake", 64, prompt)

    assert fresh.measurement == SAMPLE_MEASUREMENT.brief
    assert cached.measurement == fresh.measurement
    assert cached.from_cache is True


def test_off_selects_nothing(
    settings: Settings, reference: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        PROCESSORS, "fake", make_fake_spec("fake", RunCounter(), measurement=DISTINCT_FORMS)
    )

    result = service_measuring(settings, "off").analyze(reference, "fake", 64)

    assert result.measurement == ""


def test_brief_selects_brief(
    settings: Settings, reference: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        PROCESSORS, "fake", make_fake_spec("fake", RunCounter(), measurement=DISTINCT_FORMS)
    )

    result = service_measuring(settings, "brief").analyze(reference, "fake", 64)

    assert result.measurement == "B"


def test_full_selects_full(
    settings: Settings, reference: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        PROCESSORS, "fake", make_fake_spec("fake", RunCounter(), measurement=DISTINCT_FORMS)
    )

    result = service_measuring(settings, "full").analyze(reference, "fake", 64)

    assert result.measurement == "F"


def test_selection_applies_to_cache_hit(
    settings: Settings, reference: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        PROCESSORS, "fake", make_fake_spec("fake", RunCounter(), measurement=DISTINCT_FORMS)
    )
    service_measuring(settings, "brief").analyze(reference, "fake", 64)

    cached = service_measuring(settings, "full").analyze(reference, "fake", 64)

    assert cached.from_cache is True
    assert cached.measurement == "F"


def test_cached_png_without_measurement_is_rendered_again(
    service: AnalysisService, reference: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = RunCounter()
    spec = make_fake_spec("fake", counter, measurement=SAMPLE_MEASUREMENT)
    monkeypatch.setitem(PROCESSORS, "fake", spec)
    first = service.analyze(reference, "fake", 64)
    cache_path = next(service.settings.output_dir.rglob("fake-v1-64.png"))
    cache_path.write_bytes(image_to_png_bytes(Image.open(io.BytesIO(first.png))))

    result = service.analyze(reference, "fake", 64)

    assert result.from_cache is False
    assert result.measurement == SAMPLE_MEASUREMENT.brief
    assert png_measurement(cache_path.read_bytes()) == SAMPLE_MEASUREMENT


def test_unreadable_cache_file_is_rendered_again(service: AnalysisService, reference: str) -> None:
    first = service.analyze(reference, "canny", 64)
    cache_path = next(service.settings.output_dir.rglob("canny-v1-64.png"))
    cache_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"truncated")

    result = service.analyze(reference, "canny", 64)

    assert result.from_cache is False
    assert result.png == first.png
    assert cache_path.read_bytes() == first.png


def test_sample_reads_the_rendered_map(
    service: AnalysisService, reference: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fake map holds level 50 on its left half, so x 0.25 reads 50."""
    monkeypatch.setitem(PROCESSORS, "fake", sampled_test_spec("fake"))

    report = service.sample(reference, "fake", [(0.25, 0.5)], None, None, 64)

    assert report.analysis == "fake"
    assert [sample.value for sample in report.samples] == [[50]]


def test_sample_line_uses_the_default_count(
    service: AnalysisService, reference: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 32 default samples cross the one step of the fake map, from 50 up to 200."""
    monkeypatch.setitem(PROCESSORS, "fake", sampled_test_spec("fake"))

    report = service.sample(reference, "fake", None, (0.0, 0.5, 1.0, 0.5), None, 64)

    assert len(report.samples) == 32
    assert [change.size for change in report.changes] == [150]


def test_sample_maps_positions_through_the_crop(
    service: AnalysisService, reference: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In a crop of the right half, x 0.6 is 0.2 across the fake map, on its level 50 half."""
    monkeypatch.setitem(PROCESSORS, "fake", sampled_test_spec("fake"))
    crop = CropRegion.from_list([0.5, 0.0, 1.0, 1.0])

    report = service.sample(reference, "fake", [(0.6, 0.5)], None, None, 64, crop)

    assert [sample.value for sample in report.samples] == [[50]]


def test_sample_serves_cached_render(
    service: AnalysisService, reference: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs: list[int] = []
    monkeypatch.setitem(PROCESSORS, "fake", sampled_test_spec("fake", runs))

    service.sample(reference, "fake", [(0.25, 0.5)], None, None, 64)
    service.sample(reference, "fake", [(0.75, 0.5)], None, None, 64)

    assert runs == [64]


def test_sample_rejects_analysis_without_values(service: AnalysisService, reference: str) -> None:
    with pytest.raises(SamplingError, match="depth, normals"):
        service.sample(reference, "canny", [(0.5, 0.5)], None, None, 64)


def test_sample_rejects_points_together_with_line(
    service: AnalysisService, reference: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(PROCESSORS, "fake", sampled_test_spec("fake"))

    with pytest.raises(SamplingError, match="exactly one"):
        service.sample(reference, "fake", [(0.5, 0.5)], (0.0, 0.5, 1.0, 0.5), None, 64)


def test_sample_rejects_neither_points_nor_line(
    service: AnalysisService, reference: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(PROCESSORS, "fake", sampled_test_spec("fake"))

    with pytest.raises(SamplingError, match="exactly one"):
        service.sample(reference, "fake", None, None, None, 64)


def test_sample_rejects_count_with_points(
    service: AnalysisService, reference: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(PROCESSORS, "fake", sampled_test_spec("fake"))

    with pytest.raises(SamplingError, match="count"):
        service.sample(reference, "fake", [(0.5, 0.5)], None, 8, 64)


@pytest.fixture
def pair_of_references(settings: Settings) -> tuple[str, str]:
    """Two references of the same proportions, 128x64 and 64x32."""
    write_test_image(settings.reference_image_dir / "large.png", size=(128, 64))
    write_test_image(settings.reference_image_dir / "small.png", size=(64, 32))
    return "large.png", "small.png"


def record_detected_sizes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """Replace line detection with one that finds a single edge and records each image's size."""
    sizes: list[tuple[int, int]] = []

    def detect(
        detector: object, image: Image.Image, resolution: int
    ) -> tuple[np.ndarray, int, int]:
        sizes.append(image.size)
        return np.array([[0.0, 10.0, 60.0, 10.0]]), image.width, image.height

    monkeypatch.setattr("controlnet_mcp.analysis.detect_line_segments", detect)
    monkeypatch.setitem(PROCESSORS, "lines", make_fake_spec("lines", RunCounter()))
    return sizes


def test_compare_pairs_the_edges_of_two_images(
    service: AnalysisService, pair_of_references: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both images have one edge at the same place of a 128x64 frame, so they pair at no offset."""
    record_detected_sizes(monkeypatch)

    result = service.compare("large.png", "large.png", align="none", resolution=64)

    assert result.png[:8] == b"\x89PNG\r\n\x1a\n"
    assert "1 pair of edges" in result.measurement
    assert "offset (+0.00, +0.00)" in result.measurement
    assert result.align == "none"


def test_compare_crops_both_images_by_the_same_fractions(
    service: AnalysisService, pair_of_references: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The right half of a 128x64 image and of a 64x32 image are both square."""
    sizes = record_detected_sizes(monkeypatch)
    first, second = pair_of_references

    result = service.compare(
        first, second, align="none", resolution=64, crop=CropRegion.from_list([0.5, 0, 1, 1])
    )

    assert sizes == [(64, 64), (64, 64)]
    assert result.crop == CropRegion(0.5, 0.0, 1.0, 1.0)


def test_compare_with_plain_images_withholds_the_alignment(
    service: AnalysisService, pair_of_references: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Solid-color references have no features to fit a transform to."""
    record_detected_sizes(monkeypatch)
    first, second = pair_of_references

    result = service.compare(first, second, resolution=64)

    assert "Alignment withheld" in result.measurement
    assert Image.open(io.BytesIO(result.png)).size == (256, 64)


def test_compare_none_with_different_aspect_ratios_is_error(
    service: AnalysisService, settings: Settings
) -> None:
    write_test_image(settings.reference_image_dir / "chair.png", size=(64, 32))
    write_test_image(settings.reference_image_dir / "table.jpg", size=(16, 16))

    with pytest.raises(ComparisonError, match="proportions"):
        service.compare("chair.png", "table.jpg", align="none", resolution=64)


def test_compare_rejects_files_outside_the_directory(
    service: AnalysisService, reference: str
) -> None:
    with pytest.raises(ReferenceImageError):
        service.compare(reference, "../x.png")


def test_compare_off_reports_no_measurement(
    settings: Settings, pair_of_references: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    record_detected_sizes(monkeypatch)

    result = service_measuring(settings, "off").compare(
        "large.png", "large.png", align="none", resolution=64
    )

    assert result.measurement == ""


class StatefulDetector:
    """Stand-in for a detector that keeps per-image state between two steps of a run."""

    def __init__(self) -> None:
        self.current: tuple[int, int, int] | None = None


def make_stateful_spec(kind: str) -> ProcessorSpec:
    """A prompted processor whose run reads back state set earlier in the same run."""

    def build(model_dir: Path, device: torch.device) -> object:
        return StatefulDetector()

    def run(
        detector: object,
        image: Image.Image,
        resolution: int,
        prompt: RegionPrompt | None,
        region: CropRegion,
        options: AnalysisOptions,
    ) -> AnalysisOutput:
        assert isinstance(detector, StatefulDetector)
        detector.current = image.getpixel((0, 0))
        time.sleep(0.05)
        return AnalysisOutput(Image.new("RGB", (4, 4), detector.current))

    return ProcessorSpec(
        kind=kind,
        description="Stateful fake.",
        use_when="Use it to test concurrency.",
        checkpoints=(),
        build=build,
        run=run,
        accepts_prompt=True,
    )


def test_concurrent_analyses_do_not_share_detector_state(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    detector = StatefulDetector()
    spec = make_stateful_spec("fake")
    monkeypatch.setitem(PROCESSORS, "fake", spec)
    manager = ModelManager(settings.model_dir, torch.device("cpu"), max_loaded=1)
    monkeypatch.setattr(manager, "get", lambda requested: detector)
    service = AnalysisService(settings, manager, AnalysisCache(settings.output_dir))
    colors = {"red.png": (200, 0, 0), "blue.png": (0, 0, 200)}
    for name, color in colors.items():
        write_test_image(settings.reference_image_dir / name, size=(8, 8), color=color)
    results: dict[str, tuple[int, int, int]] = {}
    prompt = RegionPrompt.from_lists(None, [0.5, 0.5])

    def analyze(name: str) -> None:
        result = service.analyze(name, "fake", 64, prompt)
        pixel = Image.open(io.BytesIO(result.png)).getpixel((0, 0))
        results[name] = pixel  # type: ignore[assignment]

    threads = [threading.Thread(target=analyze, args=(name,)) for name in colors]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == colors
