"""Shared test helpers for the controlnet_mcp suite."""

from collections.abc import Callable, Iterable
from pathlib import Path

import pytest
from dotenv import dotenv_values
from PIL import Image

from controlnet_mcp.checkpoints import CheckpointSpec, missing_checkpoints
from controlnet_mcp.measurements import Measurement

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
