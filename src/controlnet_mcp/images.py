"""Confined access to the reference image directory.

Every filename that arrives from an MCP tool argument passes through
``resolve_reference_path`` so the server never reads outside the configured
directory and only serves supported image formats.
"""

import io
import logging
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from PIL.PngImagePlugin import PngInfo
from pydantic import BaseModel, Field

from controlnet_mcp.measurements import Measurement

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp"})

_MIME_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


class ReferenceImageError(Exception):
    """Raised when a requested reference image is outside the directory, missing, or unsupported."""


class ReferenceImageInfo(BaseModel):
    """Lightweight metadata about one reference image, as returned to agents."""

    filename: str = Field(description="File name inside the reference image directory.")
    width: int = Field(description="Image width in pixels.")
    height: int = Field(description="Image height in pixels.")
    format: str = Field(description="Image format as reported by Pillow, for example PNG.")


def resolve_reference_path(directory: Path, filename: str) -> Path:
    """Map an agent-supplied filename onto a file inside ``directory``.

    Rejects empty names, absolute paths, names containing path separators or
    parent references, paths that resolve outside the directory (including
    through symlinks), non-files, and unsupported extensions.

    Raises:
        ReferenceImageError: When any check fails.
    """
    if not filename or filename != filename.strip():
        raise ReferenceImageError(
            "Filename must be a non-empty name without surrounding whitespace."
        )
    candidate = Path(filename)
    if candidate.is_absolute() or len(candidate.parts) != 1 or candidate.name in {".", ".."}:
        raise ReferenceImageError(
            f"Filename {filename!r} must be a bare file name inside the reference image directory."
        )
    root = directory.resolve()
    resolved = (root / candidate.name).resolve()
    if not resolved.is_relative_to(root) or resolved.parent != root:
        raise ReferenceImageError(
            f"Filename {filename!r} resolves outside the reference image directory."
        )
    if resolved.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ReferenceImageError(
            f"Filename {filename!r} has an unsupported extension; supported: "
            + ", ".join(sorted(SUPPORTED_EXTENSIONS))
        )
    if not resolved.exists():
        raise ReferenceImageError(f"Reference image {filename!r} does not exist.")
    if not resolved.is_file():
        raise ReferenceImageError(f"Reference image {filename!r} is not a regular file.")
    return resolved


def list_reference_images(directory: Path) -> list[ReferenceImageInfo]:
    """Describe every image directly inside ``directory``, sorted by filename.

    An entry is listed only when ``resolve_reference_path`` would accept its
    name, so the listing and the fetch apply one rule. Files that Pillow
    cannot identify are skipped with a warning.
    """
    entries: list[ReferenceImageInfo] = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        try:
            resolved = resolve_reference_path(directory, path.name)
        except ReferenceImageError:
            continue
        try:
            with Image.open(resolved) as image:
                entries.append(
                    ReferenceImageInfo(
                        filename=path.name,
                        width=image.width,
                        height=image.height,
                        format=image.format or path.suffix.lstrip(".").upper(),
                    )
                )
        except (UnidentifiedImageError, OSError):
            logger.warning("Skipping unreadable reference image %s", path.name)
    return entries


def read_reference_bytes(directory: Path, filename: str) -> tuple[bytes, str]:
    """Return the raw bytes of a reference image and its MIME type."""
    path = resolve_reference_path(directory, filename)
    return path.read_bytes(), _MIME_TYPES[path.suffix.lower()]


def decode_reference_image(data: bytes, filename: str) -> Image.Image:
    """Decode reference image bytes to RGB.

    Args:
        data: The file contents, as returned by ``read_reference_bytes``.
        filename: Used only to name the image in the error message.

    Raises:
        ReferenceImageError: When the bytes are not a decodable image.
    """
    try:
        with Image.open(io.BytesIO(data)) as image:
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ReferenceImageError(f"Reference image {filename!r} could not be decoded.") from exc


PNG_BRIEF_KEY = "measure-brief"
PNG_FULL_KEY = "measure-full"


def image_to_png_bytes(image: Image.Image, measurement: Measurement | None = None) -> bytes:
    """Encode a PIL image as PNG bytes, storing each non-empty measurement form in a text chunk."""
    buffer = io.BytesIO()
    metadata = PngInfo()
    if measurement is not None:
        if measurement.brief:
            metadata.add_text(PNG_BRIEF_KEY, measurement.brief)
        if measurement.full:
            metadata.add_text(PNG_FULL_KEY, measurement.full)
    image.save(buffer, format="PNG", pnginfo=metadata)
    return buffer.getvalue()


def png_measurement(data: bytes) -> Measurement:
    """Return the measurement stored by ``image_to_png_bytes``.

    A form whose text chunk is absent comes back empty, so a PNG written
    without a measurement yields ``EMPTY_MEASUREMENT``.
    """
    with Image.open(io.BytesIO(data)) as image:
        text = getattr(image, "text", {})
        return Measurement(
            brief=str(text.get(PNG_BRIEF_KEY, "")),
            full=str(text.get(PNG_FULL_KEY, "")),
        )
