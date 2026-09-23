"""Tests for the analysis service: validation, caching, and processor dispatch."""

import io
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch
from PIL import Image

from conftest import write_test_image
from controlnet_mcp.analysis import AnalysisService, ResolutionError
from controlnet_mcp.cache import AnalysisCache
from controlnet_mcp.checkpoints import ZOE_CHECKPOINT, CheckpointSpec
from controlnet_mcp.config import Settings
from controlnet_mcp.model_manager import MissingCheckpointError, ModelManager
from controlnet_mcp.processors import (
    PROCESSORS,
    AnalysisOutput,
    ProcessorSpec,
    UnknownAnalysisError,
)
from controlnet_mcp.segmentation import PromptError, RegionPrompt


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
) -> ProcessorSpec:
    def build(model_dir: Path, device: torch.device) -> object:
        return object()

    def run(
        detector: object, image: Image.Image, resolution: int, prompt: RegionPrompt | None
    ) -> AnalysisOutput:
        counter.calls += 1
        counter.resolutions.append(resolution)
        counter.prompts.append(prompt)
        note = f"Region from {prompt.digest()}." if prompt is not None else ""
        return AnalysisOutput(Image.new("RGB", (resolution, resolution // 2), (0, 0, 255)), note)

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
    )


@pytest.fixture
def service(settings: Settings) -> AnalysisService:
    manager = ModelManager(settings.model_dir, torch.device("cpu"), max_loaded=1)
    return AnalysisService(settings, manager, AnalysisCache(settings.output_dir))


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
    assert list(service.settings.output_dir.rglob("fake-128.png"))


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


def test_note_survives_cache(
    service: AnalysisService, reference: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = RunCounter()
    monkeypatch.setitem(PROCESSORS, "fake", make_fake_spec("fake", counter, accepts_prompt=True))
    prompt = RegionPrompt.from_lists(None, [0.3, 0.3])

    fresh = service.analyze(reference, "fake", 64, prompt)
    cached = service.analyze(reference, "fake", 64, prompt)

    assert fresh.note == f"Region from {prompt.digest()}."
    assert cached.note == fresh.note
    assert cached.from_cache is True
