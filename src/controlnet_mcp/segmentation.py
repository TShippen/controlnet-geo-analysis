"""Prompted region segmentation: normalized box or point prompts and their rendering.

The segmenter wraps the vendored SAM predictor so an image is encoded once and
each prompt costs only a decoder pass. Prompts arrive from tool arguments as
normalized coordinates, which keeps them independent of the detect resolution.
"""

import hashlib
import logging
from dataclasses import dataclass

import numpy as np
import torch
from controlnet_aux.segment_anything.predictor import SamPredictor
from controlnet_aux.util import HWC3, resize_image
from PIL import Image

logger = logging.getLogger(__name__)

OVERLAY_COLOR = np.array([255, 80, 0], dtype=np.float32)
OVERLAY_ALPHA = 0.45
OUTLINE_COLOR = np.array([255, 255, 255], dtype=np.uint8)
PROMPT_DIGEST_LENGTH = 8


class PromptError(ValueError):
    """Raised when a region prompt is malformed, missing, or given to an analysis without one."""


@dataclass(frozen=True)
class RegionPrompt:
    """Where to segment, in normalized image coordinates with the origin at the top left."""

    box: tuple[float, float, float, float] | None = None
    point: tuple[float, float] | None = None

    @classmethod
    def from_lists(cls, box: list[float] | None, point: list[float] | None) -> "RegionPrompt":
        """Validate tool arguments and build a prompt.

        Raises:
            PromptError: When neither value is given, a list has the wrong length, a
                coordinate is outside 0 to 1, or the box is not top-left to bottom-right.
        """
        if box is None and point is None:
            raise PromptError("Segmentation needs a box [x0, y0, x1, y1] or a point [x, y].")
        validated_box: tuple[float, float, float, float] | None = None
        validated_point: tuple[float, float] | None = None
        if box is not None:
            if len(box) != 4:
                raise PromptError("box must have four values: [x0, y0, x1, y1].")
            _check_unit_range("box", box)
            x0, y0, x1, y1 = (float(value) for value in box)
            if x0 >= x1 or y0 >= y1:
                raise PromptError("box must run from the top-left corner to the bottom-right.")
            validated_box = (x0, y0, x1, y1)
        if point is not None:
            if len(point) != 2:
                raise PromptError("point must have two values: [x, y].")
            _check_unit_range("point", point)
            validated_point = (float(point[0]), float(point[1]))
        return cls(box=validated_box, point=validated_point)

    def digest(self) -> str:
        """Short stable identifier for cache file names."""
        canonical = f"box={self.box!r};point={self.point!r}"
        return hashlib.sha256(canonical.encode()).hexdigest()[:PROMPT_DIGEST_LENGTH]


def _check_unit_range(name: str, values: list[float]) -> None:
    for value in values:
        if not 0.0 <= float(value) <= 1.0:
            raise PromptError(f"{name} coordinates must be between 0 and 1; got {value}.")


class PromptedSegmenter:
    """Segments a region of an image from a box or point prompt.

    Holds the image embedding of the most recently segmented image so repeated
    prompts on the same image skip the encoder.
    """

    def __init__(self, sam_model: torch.nn.Module, device: torch.device) -> None:
        sam_model.to(device)
        sam_model.eval()
        self.predictor = SamPredictor(sam_model)
        self._embedded_digest: str | None = None

    def segment(self, image: Image.Image, prompt: RegionPrompt) -> tuple[np.ndarray, float]:
        """Return the best mask for ``prompt`` and its score.

        The mask has the shape of the image handed in.
        """
        pixels = HWC3(np.array(image, dtype=np.uint8))
        digest = hashlib.sha256(pixels.tobytes()).hexdigest()
        if digest != self._embedded_digest:
            logger.info("Encoding image for segmentation (%dx%d)", pixels.shape[1], pixels.shape[0])
            self.predictor.set_image(pixels)
            self._embedded_digest = digest
        height, width = pixels.shape[:2]
        box = None
        if prompt.box is not None:
            x0, y0, x1, y1 = prompt.box
            box = np.array([x0 * width, y0 * height, x1 * width, y1 * height])
        point_coords = None
        point_labels = None
        if prompt.point is not None:
            point_coords = np.array([[prompt.point[0] * width, prompt.point[1] * height]])
            point_labels = np.array([1])
        with torch.no_grad():
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True,
            )
        best = int(np.argmax(scores))
        return masks[best].astype(bool), float(scores[best])


def resize_for_detection(image: Image.Image, resolution: int) -> Image.Image:
    """Scale the short side to ``resolution`` with both sides rounded to multiples of 64."""
    pixels = HWC3(np.array(image.convert("RGB"), dtype=np.uint8))
    return Image.fromarray(resize_image(pixels, resolution))


def render_region_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Tint the masked pixels and draw a one-pixel outline around the region."""
    pixels = np.array(image.convert("RGB"), dtype=np.float32)
    tinted = pixels.copy()
    tinted[mask] = pixels[mask] * (1.0 - OVERLAY_ALPHA) + OVERLAY_COLOR * OVERLAY_ALPHA
    result = tinted.clip(0, 255).astype(np.uint8)
    result[_outline(mask)] = OUTLINE_COLOR
    return Image.fromarray(result)


def _outline(mask: np.ndarray) -> np.ndarray:
    """Pixels inside the mask that touch a pixel outside it."""
    padded = np.pad(mask, 1, constant_values=False)
    eroded = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:] & mask
    return mask & ~eroded


def describe_region(mask: np.ndarray) -> str:
    """Summarize a mask as an area share and a normalized bounding box."""
    if not mask.any():
        return "No region found."
    height, width = mask.shape
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    area = 100.0 * mask.sum() / mask.size
    x0, x1 = cols[0] / width, (cols[-1] + 1) / width
    y0, y1 = rows[0] / height, (rows[-1] + 1) / height
    return (
        f"Region covers {area:.1f}% of the image; bounding box x {x0:.2f} to {x1:.2f}, "
        f"y {y0:.2f} to {y1:.2f} (normalized, origin top-left)."
    )
