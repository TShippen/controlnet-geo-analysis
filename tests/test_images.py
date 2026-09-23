"""Tests for confined reference image access."""

import io
from pathlib import Path

import pytest
from PIL import Image

from conftest import SAMPLE_MEASUREMENT, write_test_image
from controlnet_mcp.images import (
    ReferenceImageError,
    decode_reference_image,
    image_to_png_bytes,
    list_reference_images,
    png_measurement,
    read_reference_bytes,
    resolve_reference_path,
)
from controlnet_mcp.measurements import EMPTY_MEASUREMENT


@pytest.fixture
def reference_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "references"
    directory.mkdir()
    return directory


def test_list_returns_supported_images_with_metadata(reference_dir: Path) -> None:
    write_test_image(reference_dir / "b.png", size=(32, 16))
    write_test_image(reference_dir / "a.png", size=(8, 8))
    (reference_dir / "notes.txt").write_text("not an image")

    entries = list_reference_images(reference_dir)

    assert [entry.filename for entry in entries] == ["a.png", "b.png"]
    assert (entries[0].width, entries[0].height, entries[0].format) == (8, 8, "PNG")
    assert (entries[1].width, entries[1].height) == (32, 16)


def test_list_skips_subdirectories(reference_dir: Path) -> None:
    nested = reference_dir / "nested"
    nested.mkdir()
    write_test_image(nested / "inner.png")

    assert list_reference_images(reference_dir) == []


def test_list_skips_symlink_escape(reference_dir: Path, tmp_path: Path) -> None:
    outside = write_test_image(tmp_path / "outside.png")
    (reference_dir / "link.png").symlink_to(outside)
    write_test_image(reference_dir / "inside.png")

    assert [entry.filename for entry in list_reference_images(reference_dir)] == ["inside.png"]


def test_list_skips_unreadable_files(reference_dir: Path) -> None:
    (reference_dir / "broken.png").write_bytes(b"not really a png")
    write_test_image(reference_dir / "ok.png")

    assert [entry.filename for entry in list_reference_images(reference_dir)] == ["ok.png"]


def test_resolve_rejects_traversal(reference_dir: Path, tmp_path: Path) -> None:
    write_test_image(tmp_path / "secret.png")

    with pytest.raises(ReferenceImageError):
        resolve_reference_path(reference_dir, "../secret.png")


def test_resolve_rejects_absolute_path(reference_dir: Path) -> None:
    with pytest.raises(ReferenceImageError):
        resolve_reference_path(reference_dir, "/etc/passwd")


def test_resolve_rejects_unsupported_extension(reference_dir: Path) -> None:
    (reference_dir / "notes.txt").write_text("x")

    with pytest.raises(ReferenceImageError, match=r"\.png"):
        resolve_reference_path(reference_dir, "notes.txt")


def test_resolve_rejects_missing_file(reference_dir: Path) -> None:
    with pytest.raises(ReferenceImageError, match="does not exist"):
        resolve_reference_path(reference_dir, "missing.png")


def test_resolve_rejects_symlink_escape(reference_dir: Path, tmp_path: Path) -> None:
    outside = write_test_image(tmp_path / "outside.png")
    (reference_dir / "link.png").symlink_to(outside)

    with pytest.raises(ReferenceImageError, match="outside"):
        resolve_reference_path(reference_dir, "link.png")


def test_resolve_accepts_plain_file(reference_dir: Path) -> None:
    write_test_image(reference_dir / "photo.jpg")

    assert (
        resolve_reference_path(reference_dir, "photo.jpg")
        == (reference_dir / "photo.jpg").resolve()
    )


def test_read_reference_bytes_returns_mime(reference_dir: Path) -> None:
    write_test_image(reference_dir / "photo.jpg")

    data, mime = read_reference_bytes(reference_dir, "photo.jpg")

    assert mime == "image/jpeg"
    assert data[:2] == b"\xff\xd8"


def test_decode_reference_image_converts_to_rgb(reference_dir: Path) -> None:
    path = write_test_image(reference_dir / "alpha.png", color=(1, 2, 3, 128), mode="RGBA")

    image = decode_reference_image(path.read_bytes(), "alpha.png")

    assert image.mode == "RGB"


def test_decode_reference_image_rejects_garbage() -> None:
    with pytest.raises(ReferenceImageError, match="decoded"):
        decode_reference_image(b"not an image", "bad.png")


def test_image_to_png_bytes_roundtrip() -> None:
    image = Image.new("RGB", (20, 10), (0, 255, 0))

    data = image_to_png_bytes(image)

    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert Image.open(io.BytesIO(data)).size == (20, 10)


def test_png_measurement_roundtrip() -> None:
    image = Image.new("RGB", (4, 4))

    measured = image_to_png_bytes(image, SAMPLE_MEASUREMENT)
    unmeasured = image_to_png_bytes(image)

    assert png_measurement(measured) == SAMPLE_MEASUREMENT
    assert png_measurement(unmeasured) == EMPTY_MEASUREMENT
