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

from conftest import two_point_test_segments
from controlnet_mcp.checkpoints import CheckpointSpec
from controlnet_mcp.measurements import BEYOND_RANGE_MAX, EMPTY_MEASUREMENT
from controlnet_mcp.processors import (
    ANALYSIS_KINDS,
    DEFAULT_OPTIONS,
    PROCESSORS,
    SAMPLED_KINDS,
    AnalysisOptions,
    AnalysisOutput,
    ProcessorSpec,
    UnknownAnalysisError,
    get_processor,
)
from controlnet_mcp.regions import FULL_IMAGE, CropRegion
from controlnet_mcp.segmentation import RegionPrompt

LEARNED_KINDS = ("depth", "normals", "lineart", "lines", "segments")

BRIEF_LIMIT = 160


def run_test_processor(
    spec: ProcessorSpec,
    detector: object,
    image: Image.Image,
    resolution: int,
    prompt: RegionPrompt | None = None,
    region: CropRegion = FULL_IMAGE,
    options: AnalysisOptions = DEFAULT_OPTIONS,
) -> AnalysisOutput:
    """Run a processor, on the whole image with default options unless told otherwise."""
    return spec.run(detector, image, resolution, prompt, region, options)


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


def box_scene_image() -> Image.Image:
    """A flat-shaded cube on a floor, lit from the upper left, at 640x480.

    The front face points at the camera, the top face points up, and the right
    face points to the right of the image, so each face has one known normal.
    """
    image = Image.new("RGB", (640, 480), (234, 234, 240))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 260, 640, 480), fill=(170, 160, 150))
    draw.rectangle((200, 200, 400, 400), fill=(150, 60, 50))
    draw.polygon([(200, 200), (280, 140), (480, 140), (400, 200)], fill=(210, 110, 95))
    draw.polygon([(400, 200), (480, 140), (480, 340), (400, 400)], fill=(100, 38, 33))
    return image


def median_color(image: Image.Image, center: tuple[int, int]) -> tuple[int, int, int]:
    """Median RGB of a 9x9 patch around a point given in 640x480 scene coordinates."""
    pixels = np.array(image)
    height, width = pixels.shape[:2]
    x = round(center[0] * width / 640)
    y = round(center[1] * height / 480)
    patch = pixels[y - 4 : y + 5, x - 4 : x + 5].reshape(-1, 3)
    red, green, blue = np.median(patch, axis=0)
    return int(red), int(green), int(blue)


def edge_share(brief: str) -> float:
    """The percentage an edge measurement reports, as a number."""
    match = re.search(r"([\d.]+)%", brief)
    assert match is not None, f"No percentage in {brief!r}"
    return float(match.group(1))


def test_registry_covers_all_kinds() -> None:
    assert set(PROCESSORS) == set(ANALYSIS_KINDS)
    assert all(spec.kind == kind for kind, spec in PROCESSORS.items())


def test_every_processor_has_use_when() -> None:
    assert all(spec.use_when for spec in PROCESSORS.values())


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

    output = run_test_processor(spec, detector, structured_test_image((200, 100)), 128)

    assert output.image.mode == "RGB"
    assert output.image.size == (256, 128)


def test_canny_run_measures_edges() -> None:
    spec = get_processor("canny")
    detector = spec.build(Path("/nonexistent"), torch.device("cpu"))

    output = run_test_processor(spec, detector, structured_test_image((200, 100)), 128)

    assert output.measurement != EMPTY_MEASUREMENT
    assert "%" in output.measurement.brief
    assert len(output.measurement.brief) < BRIEF_LIMIT


def test_canny_measures_the_drawn_edges_not_the_ground() -> None:
    """Edges are the minority of a canny output, so a measured majority means inverted polarity."""
    spec = get_processor("canny")
    detector = spec.build(Path("/nonexistent"), torch.device("cpu"))

    output = run_test_processor(spec, detector, structured_test_image((200, 100)), 128)

    assert 0 < edge_share(output.measurement.brief) < 50


def test_lines_version_is_two() -> None:
    assert PROCESSORS["lines"].version == "2"


def test_depth_version_is_two() -> None:
    """The bump retires cached PNGs whose stored measurement counts black as far."""
    assert PROCESSORS["depth"].version == "2"


def test_segments_version_is_two() -> None:
    """The bump retires cached box results chosen from three candidates instead of one mask."""
    assert PROCESSORS["segments"].version == "2"


def test_lineart_version_is_two() -> None:
    """The bump retires cached PNGs whose stored measurement has the inverted polarity."""
    assert PROCESSORS["lineart"].version == "2"


def test_normals_version_is_three() -> None:
    """The bump retires cached PNGs whose stored faces join separate regions into one."""
    assert PROCESSORS["normals"].version == "3"


def test_sampled_kinds_are_depth_and_normals() -> None:
    assert SAMPLED_KINDS == ("depth", "normals")


def test_every_sampled_analysis_describes_its_values() -> None:
    sampled = [spec for spec in PROCESSORS.values() if spec.read_values is not None]

    assert all(spec.values_description for spec in sampled)


def test_lines_run_draws_and_measures_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    segments = np.array([[0.0, 0.0, 60.0, 0.0], [10.0, 10.0, 10.0, 50.0]])
    monkeypatch.setattr("controlnet_mcp.processors.pred_lines", lambda *args: segments)

    output = run_test_processor(
        get_processor("lines"), MLSDdetector(object()), structured_test_image((128, 128)), 128
    )

    assert "2 straight edges" in output.measurement.brief
    assert np.array(output.image)[0, 0].tolist() == [255, 255, 255]
    assert np.array(output.image)[64, 64].tolist() == [0, 0, 0]


def test_perspective_shares_the_lines_detector() -> None:
    assert PROCESSORS["perspective"].detector_key == PROCESSORS["lines"].detector_key


def test_perspective_run_colors_families(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two converging sets are two groups, the first drawn red and the second green."""
    segments = two_point_test_segments()
    monkeypatch.setattr("controlnet_mcp.processors.pred_lines", lambda *args: segments)

    output = run_test_processor(
        get_processor("perspective"),
        MLSDdetector(object()),
        structured_test_image((512, 512)),
        512,
    )

    colors = {tuple(color) for color in np.array(output.image).reshape(-1, 3).tolist()}
    assert (255, 0, 0) in colors
    assert (0, 255, 0) in colors
    assert "vanishes at" in output.measurement.brief


def test_perspective_run_withholds_the_camera_for_a_crop(monkeypatch: pytest.MonkeyPatch) -> None:
    segments = two_point_test_segments()
    monkeypatch.setattr("controlnet_mcp.processors.pred_lines", lambda *args: segments)

    output = run_test_processor(
        get_processor("perspective"),
        MLSDdetector(object()),
        structured_test_image((512, 512)),
        512,
        region=CropRegion(0.5, 0.0, 1.0, 1.0),
    )

    assert "Camera estimate withheld: the image is a crop" in output.measurement.brief


def test_lines_run_reports_nothing_when_no_segments_are_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``pred_lines`` indexes columns of an empty result, so finding nothing raises IndexError."""

    def raise_index_error(*args: object) -> np.ndarray:
        raise IndexError("too many indices for array")

    monkeypatch.setattr("controlnet_mcp.processors.pred_lines", raise_index_error)

    output = run_test_processor(
        get_processor("lines"), MLSDdetector(object()), structured_test_image((128, 128)), 128
    )

    assert output.measurement.brief == "No straight edges found."
    assert not np.array(output.image).any()


def test_lines_run_maps_endpoints_through_region(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lines found in a crop of the lower right quarter report full-image positions."""
    segments = np.array([[0.0, 0.0, 128.0, 0.0]])
    monkeypatch.setattr("controlnet_mcp.processors.pred_lines", lambda *args: segments)

    output = run_test_processor(
        get_processor("lines"),
        MLSDdetector(object()),
        structured_test_image((128, 128)),
        128,
        region=CropRegion(0.5, 0.5, 1.0, 1.0),
    )

    assert "(0.50,0.50)-(1.00,0.50)" in output.measurement.brief


def test_lines_long_drops_segments_shorter_than_the_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a 128x128 detection the cutoff is 7.68 pixels: 60 stays, 5 goes."""
    segments = np.array([[0.0, 0.0, 60.0, 0.0], [0.0, 64.0, 5.0, 64.0]])
    monkeypatch.setattr("controlnet_mcp.processors.pred_lines", lambda *args: segments)

    output = run_test_processor(
        get_processor("lines"),
        MLSDdetector(object()),
        structured_test_image((128, 128)),
        128,
        options=AnalysisOptions(line_length="long"),
    )

    assert output.measurement.brief.startswith("1 long straight edge;")
    assert np.array(output.image)[64, 2].tolist() == [0, 0, 0]


def test_lines_long_with_nothing_long_enough(monkeypatch: pytest.MonkeyPatch) -> None:
    segments = np.array([[0.0, 64.0, 5.0, 64.0]])
    monkeypatch.setattr("controlnet_mcp.processors.pred_lines", lambda *args: segments)

    output = run_test_processor(
        get_processor("lines"),
        MLSDdetector(object()),
        structured_test_image((128, 128)),
        128,
        options=AnalysisOptions(line_length="long"),
    )

    assert output.measurement.brief == "No long straight edges found."


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
    output = run_test_processor(spec, detector, structured_test_image((128, 128)), 128, prompt)

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
def test_normals_encode_facing_direction_in_camera_space(
    installed_checkpoints: Callable[[Iterable[CheckpointSpec]], Path],
) -> None:
    """Pins the color encoding the agent-facing normals text describes.

    Each channel is centered on 128. Red is high for a face turned to the image
    left and low for one turned right, green is high for a face turned up, and
    blue is high for a face turned toward the camera.
    """
    spec = get_processor("normals")
    detector = spec.build(installed_checkpoints(spec.checkpoints), torch.device("cpu"))

    rendered = run_test_processor(spec, detector, box_scene_image(), 256).image

    front_red, front_green, front_blue = median_color(rendered, (300, 300))
    assert front_blue > 230
    assert 100 <= front_red <= 160
    assert 100 <= front_green <= 160
    assert median_color(rendered, (340, 170))[1] > 220
    right_red, _, right_blue = median_color(rendered, (440, 270))
    assert right_red < 100
    assert right_red < front_red - 30
    assert right_blue > 150


@pytest.mark.slow
@pytest.mark.integration
def test_depth_clips_the_farthest_share_to_black(
    installed_checkpoints: Callable[[Iterable[CheckpointSpec]], Path],
) -> None:
    """Pins that the depth rendering stretches each image between percentiles.

    Everything at or beyond the 85th-percentile depth renders black, so any
    scene has a solid black far region, which the agent-facing text states.
    """
    spec = get_processor("depth")
    detector = spec.build(installed_checkpoints(spec.checkpoints), torch.device("cpu"))

    rendered = run_test_processor(spec, detector, box_scene_image(), 256).image

    gray = np.array(rendered.convert("L"))
    assert float((gray <= BEYOND_RANGE_MAX).mean()) >= 0.14


@pytest.mark.slow
@pytest.mark.integration
def test_lines_on_an_image_without_straight_edges(
    installed_checkpoints: Callable[[Iterable[CheckpointSpec]], Path],
) -> None:
    """Pins what the real detector does when it finds nothing, which is to raise IndexError."""
    spec = get_processor("lines")
    detector = spec.build(installed_checkpoints(spec.checkpoints), torch.device("cpu"))

    blank = Image.new("RGB", (128, 128), (120, 120, 120))

    output = run_test_processor(spec, detector, blank, 128)

    assert output.measurement.brief == "No straight edges found."
