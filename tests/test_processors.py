"""Tests for the processor registry.

The learned detectors are exercised only in the slow, integration-marked
tests, which skip unless the checkpoints named by their spec are present
under the configured ``MODEL_DIR``.
"""

import re
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pytest
import torch
from controlnet_aux import MLSDdetector
from PIL import Image, ImageDraw

from controlnet_mcp.checkpoints import CheckpointSpec
from controlnet_mcp.measurements import EMPTY_MEASUREMENT
from controlnet_mcp.processors import (
    ANALYSIS_KINDS,
    PROCESSORS,
    UnknownAnalysisError,
    get_processor,
)
from controlnet_mcp.segmentation import RegionPrompt

LEARNED_KINDS = ("depth", "normals", "lineart", "lines", "segments")

BRIEF_LIMIT = 160


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


def edge_share(brief: str) -> float:
    """The percentage an edge measurement reports, as a number."""
    match = re.search(r"([\d.]+)%", brief)
    assert match is not None, f"No percentage in {brief!r}"
    return float(match.group(1))


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


def test_canny_run_measures_edges() -> None:
    spec = get_processor("canny")
    detector = spec.build(Path("/nonexistent"), torch.device("cpu"))

    output = spec.run(detector, structured_test_image((200, 100)), 128, None)

    assert output.measurement != EMPTY_MEASUREMENT
    assert "%" in output.measurement.brief
    assert len(output.measurement.brief) < BRIEF_LIMIT


def test_canny_measures_the_drawn_edges_not_the_ground() -> None:
    """Edges are the minority of a canny output, so a measured majority means inverted polarity."""
    spec = get_processor("canny")
    detector = spec.build(Path("/nonexistent"), torch.device("cpu"))

    output = spec.run(detector, structured_test_image((200, 100)), 128, None)

    assert 0 < edge_share(output.measurement.brief) < 50


def test_lines_version_is_two() -> None:
    assert PROCESSORS["lines"].version == "2"


def test_lineart_version_is_two() -> None:
    """The bump retires cached PNGs whose stored measurement has the inverted polarity."""
    assert PROCESSORS["lineart"].version == "2"


def test_lines_run_draws_and_measures_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    segments = np.array([[0.0, 0.0, 60.0, 0.0], [10.0, 10.0, 10.0, 50.0]])
    monkeypatch.setattr("controlnet_mcp.processors.pred_lines", lambda *args: segments)

    output = get_processor("lines").run(
        MLSDdetector(object()), structured_test_image((128, 128)), 128, None
    )

    assert "2 straight edges" in output.measurement.brief
    assert np.array(output.image)[0, 0].tolist() == [255, 255, 255]
    assert np.array(output.image)[64, 64].tolist() == [0, 0, 0]


def test_lines_run_reports_nothing_when_no_segments_are_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``pred_lines`` indexes columns of an empty result, so finding nothing raises IndexError."""

    def raise_index_error(*args: object) -> np.ndarray:
        raise IndexError("too many indices for array")

    monkeypatch.setattr("controlnet_mcp.processors.pred_lines", raise_index_error)

    output = get_processor("lines").run(
        MLSDdetector(object()), structured_test_image((128, 128)), 128, None
    )

    assert output.measurement.brief == "No straight edges found."
    assert not np.array(output.image).any()


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
    assert output.measurement.brief
    assert len(output.measurement.brief) < BRIEF_LIMIT
    if kind == "lines":
        assert output.measurement.brief[0].isdigit()
    if kind == "lineart":
        # Strokes are the minority of the drawing; a measured majority means inverted polarity.
        assert 0 < edge_share(output.measurement.brief) < 50


@pytest.mark.slow
@pytest.mark.integration
def test_lines_on_an_image_without_straight_edges(
    installed_checkpoints: Callable[[Iterable[CheckpointSpec]], Path],
) -> None:
    """Pins what the real detector does when it finds nothing, which is to raise IndexError."""
    spec = get_processor("lines")
    detector = spec.build(installed_checkpoints(spec.checkpoints), torch.device("cpu"))

    output = spec.run(detector, Image.new("RGB", (128, 128), (120, 120, 120)), 128, None)

    assert output.measurement.brief == "No straight edges found."
