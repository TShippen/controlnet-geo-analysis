"""Shared test helpers for the controlnet_mcp suite."""

from pathlib import Path

import pytest
from PIL import Image


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
