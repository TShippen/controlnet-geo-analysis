"""Tests for the analysis service: validation, caching, and processor dispatch."""

import dataclasses
import io
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch
from PIL import Image

from conftest import SAMPLE_MEASUREMENT, write_test_image
from controlnet_mcp.analysis import AnalysisService, ResolutionError
from controlnet_mcp.cache import AnalysisCache
from controlnet_mcp.checkpoints import ZOE_CHECKPOINT, CheckpointSpec
from controlnet_mcp.config import MeasurementSetting, Settings
from controlnet_mcp.images import image_to_png_bytes, png_measurement
from controlnet_mcp.measurements import Measurement
from controlnet_mcp.model_manager import MissingCheckpointError, ModelManager
from controlnet_mcp.processors import (
    PROCESSORS,
    AnalysisOutput,
    ProcessorSpec,
    UnknownAnalysisError,
)
from controlnet_mcp.segmentation import PromptError, RegionPrompt

DISTINCT_FORMS = Measurement(brief="B", full="F")


@dataclass
class RunCounter:
    """Records how many times a fake processor ran."""

    calls: int = 0
    resolutions: list[int] = field(default_factory=list)
    prompts: list[RegionPrompt | None] = field(default_factory=list)


def make_fake_spec(
    kind: str,
    counter: RunCounter,
    checkpoints: tuple[CheckpointSpec, ...] = (),
    accepts_prompt: bool = False,
    measurement: Measurement = SAMPLE_MEASUREMENT,
) -> ProcessorSpec:
    """A processor that renders a plain image and reports ``measurement``.

    The measurement defaults to a non-empty one because every real processor
    measures its output. A result without one is still written to the cache,
    but it is never served from it.
    """

    def build(model_dir: Path, device: torch.device) -> object:
        return object()

    def run(
        detector: object, image: Image.Image, resolution: int, prompt: RegionPrompt | None
    ) -> AnalysisOutput:
        counter.calls += 1
        counter.resolutions.append(resolution)
        counter.prompts.append(prompt)
        rendered = Image.new("RGB", (resolution, resolution // 2), (0, 0, 255))
        return AnalysisOutput(rendered, measurement)

    return ProcessorSpec(
        kind=kind,
        description=f"Fake {kind} analysis.",
        checkpoints=checkpoints,
        build=build,
        run=run,
        accepts_prompt=accepts_prompt,
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


class StatefulDetector:
    """Stand-in for a detector that keeps per-image state between two steps of a run."""

    def __init__(self) -> None:
        self.current: tuple[int, int, int] | None = None


def make_stateful_spec(kind: str) -> ProcessorSpec:
    """A prompted processor whose run reads back state set earlier in the same run."""

    def build(model_dir: Path, device: torch.device) -> object:
        return StatefulDetector()

    def run(
        detector: object, image: Image.Image, resolution: int, prompt: RegionPrompt | None
    ) -> AnalysisOutput:
        assert isinstance(detector, StatefulDetector)
        detector.current = image.getpixel((0, 0))
        time.sleep(0.05)
        return AnalysisOutput(Image.new("RGB", (4, 4), detector.current))

    return ProcessorSpec(
        kind=kind,
        description="Stateful fake.",
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
