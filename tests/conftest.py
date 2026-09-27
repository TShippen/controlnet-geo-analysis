"""Shared test helpers for the controlnet_mcp suite."""

from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pytest
import torch
from dotenv import dotenv_values
from PIL import Image

from controlnet_mcp.checkpoints import CheckpointSpec, missing_checkpoints
from controlnet_mcp.measurements import Measurement
from controlnet_mcp.processors import (
    PROCESSORS,
    AnalysisOptions,
    AnalysisOutput,
    ProcessorSpec,
    read_depth_values,
)
from controlnet_mcp.regions import CropRegion
from controlnet_mcp.segmentation import RegionPrompt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SAMPLE_MEASUREMENT = Measurement(
    brief="Region covers 25.0% of the image.",
    full="Region covers 25.0% of the image; centroid (0.38, 0.50).",
)
"""A measurement in the shape ``measure_mask`` produces, for tests that only need one."""


def write_test_image(
    path: Path,
    size: tuple[int, int] = (32, 32),
    color: tuple[int, int, int] | tuple[int, int, int, int] = (200, 30, 30),
    mode: str = "RGB",
) -> Path:
    """Save a solid-color image at ``path`` and return the path.

    The format is inferred from the file extension by Pillow.
    """
    image = Image.new(mode, size, color)
    image.save(path)
    return path


PERSPECTIVE_TEST_FRAME = 512
"""Side in pixels of the square frame the perspective test scenes are built in."""

RIGHT_TEST_POINT = (948.8, 256.0)
LEFT_TEST_POINT = (25.1, 256.0)
"""Vanishing points of a camera with focal length 400 turned 30 degrees from a wall.

With the optical center at (256, 256), the two perpendicular horizontal
directions vanish at 256 + 400 * tan(60°) = 948.8 and 256 - 400 * tan(30°) = 25.1.
"""

PERSPECTIVE_TEST_ANCHORS = tuple(
    (x, y) for y in (60.0, 440.0) for x in (300.0, 360.0, 420.0, 480.0)
)
"""Midpoints for converging test segments, well above and below the row of the test points."""


def converging_test_segments(point: tuple[float, float]) -> np.ndarray:
    """Eight segments centered on the test anchors, each lying on a line through ``point``.

    Their lengths run 60, 68, and so on up to 116 pixels.
    """
    segments = []
    for index, (x, y) in enumerate(PERSPECTIVE_TEST_ANCHORS):
        toward = np.array([point[0] - x, point[1] - y])
        half = (30.0 + 4.0 * index) * toward / np.linalg.norm(toward)
        segments.append([x - half[0], y - half[1], x + half[0], y + half[1]])
    return np.array(segments)


def vertical_test_segments() -> np.ndarray:
    """Eight exactly vertical segments 100 pixels long, at x 60, 120, and so on up to 480."""
    return np.array([[x, 200.0, x, 300.0] for x in np.arange(60.0, 481.0, 60.0)])


def two_point_test_segments() -> np.ndarray:
    """Two sets of eight segments converging on the right and the left test points."""
    return np.vstack(
        [converging_test_segments(RIGHT_TEST_POINT), converging_test_segments(LEFT_TEST_POINT)]
    )


def camera_scene_test_segments() -> np.ndarray:
    """The two-point scene plus verticals that stay parallel: a level camera, focal length 400."""
    return np.vstack([two_point_test_segments(), vertical_test_segments()])


def sampled_test_spec(kind: str, runs: list[int] | None = None) -> ProcessorSpec:
    """A processor that renders a depth map and can be sampled with the real depth reader.

    The map is twice as wide as it is tall, with level 50 on its left half and
    level 200 on its right half.

    Args:
        kind: The analysis name the spec answers to.
        runs: When given, each run appends its resolution, so a test can count runs.
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
        if runs is not None:
            runs.append(resolution)
        rendered = Image.new("RGB", (2 * resolution, resolution), (50, 50, 50))
        rendered.paste((200, 200, 200), (resolution, 0, 2 * resolution, resolution))
        return AnalysisOutput(rendered, SAMPLE_MEASUREMENT)

    return ProcessorSpec(
        kind=kind,
        description=f"Fake {kind} analysis.",
        use_when=f"Use fake {kind} for tests.",
        checkpoints=(),
        build=build,
        run=run,
        read_values=read_depth_values,
        values_description="Fake levels.",
    )


def fake_test_line_detection(
    monkeypatch: pytest.MonkeyPatch, edges: int = 1
) -> list[tuple[int, int]]:
    """Replace the line detection a comparison runs, which needs a checkpoint, with a fake.

    The fake finds ``edges`` horizontal edges in every image, 60 pixels long
    on rows 1, 2, and so on, the same in every image.

    Returns:
        A list that receives the size of each image the fake is given.
    """
    sizes: list[tuple[int, int]] = []
    found = np.array([[0.0, float(row), 60.0, float(row)] for row in range(1, edges + 1)])

    def detect(
        detector: object, image: Image.Image, resolution: int
    ) -> tuple[np.ndarray, int, int]:
        sizes.append(image.size)
        return found, image.width, image.height

    monkeypatch.setattr("controlnet_mcp.analysis.detect_line_segments", detect)
    monkeypatch.setitem(PROCESSORS, "lines", sampled_test_spec("lines"))
    return sizes


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on asyncio via the anyio pytest plugin."""
    return "asyncio"


@pytest.fixture
def installed_checkpoints() -> Callable[[Iterable[CheckpointSpec]], Path]:
    """Return a function that yields the model directory named by the project .env.

    The .env is read without exporting it into the process, so the fast suite's
    settings never depend on a developer's local configuration. A relative
    ``MODEL_DIR`` is taken relative to the project root, where the .env lives,
    as the server does. Calling the returned function skips the test when
    ``MODEL_DIR`` is unset or any of the given checkpoints is absent.
    """
    value = dotenv_values(PROJECT_ROOT / ".env").get("MODEL_DIR")

    def require(specs: Iterable[CheckpointSpec]) -> Path:
        if not value:
            pytest.skip("MODEL_DIR is not configured in the project .env")
        model_dir = PROJECT_ROOT / Path(value).expanduser()
        missing = missing_checkpoints(model_dir, specs)
        if missing:
            pytest.skip("Missing checkpoints: " + ", ".join(spec.filename for spec in missing))
        return model_dir

    return require
