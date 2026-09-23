"""Tests for the processor registry.

The learned detectors are exercised only in the slow, integration-marked
tests, which skip unless the checkpoints named by their spec are present
under the configured ``MODEL_DIR``.
"""

from collections.abc import Callable, Iterable
from pathlib import Path

import pytest
import torch
from PIL import Image, ImageDraw

from controlnet_mcp.checkpoints import CheckpointSpec
from controlnet_mcp.processors import (
    ANALYSIS_KINDS,
    PROCESSORS,
    UnknownAnalysisError,
    get_processor,
)
from controlnet_mcp.segmentation import RegionPrompt

LEARNED_KINDS = ("depth", "normals", "lineart", "lines", "segments")


def structured_test_image(size: tuple[int, int]) -> Image.Image:
    """A gradient with a filled rectangle, so edge, line, and region detectors find structure."""
    width, height = size
    image = Image.new("RGB", size)
    pixels = image.load()
    assert pixels is not None
    for x in range(width):
        value = int(255 * x / max(width - 1, 1))
        for y in range(height):
            pixels[x, y] = (value, value, 255 - value)
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (width // 4, height // 4, width * 3 // 4, height * 3 // 4),
        fill=(250, 250, 250),
        outline=(10, 10, 10),
        width=2,
    )
    return image


def test_registry_covers_all_kinds() -> None:
    assert set(PROCESSORS) == set(ANALYSIS_KINDS)
    assert all(spec.kind == kind for kind, spec in PROCESSORS.items())


def test_canny_requires_no_model() -> None:
    assert PROCESSORS["canny"].requires_model is False
    assert all(PROCESSORS[kind].requires_model for kind in LEARNED_KINDS)


def test_only_segments_accepts_prompt() -> None:
    assert [kind for kind, spec in PROCESSORS.items() if spec.accepts_prompt] == ["segments"]


def test_unknown_kind_raises() -> None:
    with pytest.raises(UnknownAnalysisError, match="pose"):
        get_processor("pose")


def test_canny_run_produces_rgb_at_resolution() -> None:
    spec = get_processor("canny")
    detector = spec.build(Path("/nonexistent"), torch.device("cpu"))

    output = spec.run(detector, structured_test_image((200, 100)), 128, None)

    assert output.image.mode == "RGB"
    assert output.image.size == (256, 128)
    assert output.note == ""


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("kind", LEARNED_KINDS)
def test_learned_processor_produces_rgb_at_resolution(
    kind: str, installed_checkpoints: Callable[[Iterable[CheckpointSpec]], Path]
) -> None:
    spec = get_processor(kind)
    model_dir = installed_checkpoints(spec.checkpoints)

    detector = spec.build(model_dir, torch.device("cpu"))
    prompt = RegionPrompt.from_lists([0.2, 0.2, 0.8, 0.8], None) if spec.accepts_prompt else None
    output = spec.run(detector, structured_test_image((128, 128)), 128, prompt)

    assert output.image.mode == "RGB"
    assert output.image.size == (128, 128)
    assert bool(output.note) is spec.accepts_prompt
