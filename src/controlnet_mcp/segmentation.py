"""Prompted region segmentation: normalized box or point prompts and their rendering.

The segmenter wraps the vendored SAM predictor so an image is encoded once and
each prompt costs only a decoder pass. Prompts arrive from tool arguments as
normalized coordinates, which keeps them independent of the detect resolution.
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from controlnet_aux.segment_anything.predictor import SamPredictor
from controlnet_aux.util import HWC3, resize_image
from PIL import Image

logger = logging.getLogger(__name__)

Extent = Literal["best", "largest", "smallest"]

OVERLAY_COLOR = np.array([255, 80, 0], dtype=np.float32)
OVERLAY_ALPHA = 0.45
OUTLINE_COLOR = np.array([255, 255, 255], dtype=np.uint8)
PROMPT_DIGEST_LENGTH = 8
PROMPT_DECIMALS = 4


class PromptError(ValueError):
    """Raised when a region prompt is malformed, missing, or given to an analysis without one."""


@dataclass(frozen=True)
class RegionPrompt:
    """Where to segment, in normalized image coordinates with the origin at the top left.

    A box or a point says what to include, and each exclude point marks a spot
    that must stay outside the region. A lone point is ambiguous between a
    whole object and its parts, so only that prompt yields several candidate
    masks, and ``extent`` says which one to keep. Every other prompt yields one.
    """

    box: tuple[float, float, float, float] | None = None
    point: tuple[float, float] | None = None
    exclude: tuple[tuple[float, float], ...] = ()
    extent: Extent = "best"

    @classmethod
    def from_lists(
        cls,
        box: list[float] | None,
        point: list[float] | None,
        exclude: list[list[float]] | None = None,
        extent: Extent | None = None,
    ) -> "RegionPrompt":
        """Validate tool arguments and build a prompt.

        Coordinates are rounded to ``PROMPT_DECIMALS`` places, which also maps
        negative zero to zero, so prompts that differ only by float noise share
        one cache entry.

        Raises:
            PromptError: When neither box nor point is given, a list has the wrong
                length, a coordinate is outside 0 to 1, the box is not top-left to
                bottom-right, or an extent other than best accompanies anything
                but a lone point.
        """
        if box is None and point is None:
            raise PromptError("Segmentation needs a box [x0, y0, x1, y1] or a point [x, y].")
        validated_box: tuple[float, float, float, float] | None = None
        validated_point: tuple[float, float] | None = None
        if box is not None:
            if len(box) != 4:
                raise PromptError("box must have four values: [x0, y0, x1, y1].")
            _check_unit_range("box", box)
            x0, y0, x1, y1 = (_normalize(value) for value in box)
            if x0 >= x1 or y0 >= y1:
                raise PromptError("box must run from the top-left corner to the bottom-right.")
            validated_box = (x0, y0, x1, y1)
        if point is not None:
            validated_point = _validated_point("point", point)
        excluded = tuple(_validated_point("exclude point", spot) for spot in exclude or [])
        chosen: Extent = extent or "best"
        if chosen != "best" and (validated_box is not None or excluded):
            raise PromptError(
                "extent applies only to a lone point; a box or exclude points already "
                "pin down one region."
            )
        return cls(box=validated_box, point=validated_point, exclude=excluded, extent=chosen)

    @property
    def is_lone_point(self) -> bool:
        """Whether the prompt is one point with no box and no exclusions."""
        return self.box is None and not self.exclude

    def digest(self) -> str:
        """Short stable identifier for cache file names."""
        canonical = (
            f"box={self.box!r};point={self.point!r};exclude={self.exclude!r};"
            f"extent={self.extent}"
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:PROMPT_DIGEST_LENGTH]


def _validated_point(name: str, values: list[float]) -> tuple[float, float]:
    """Check one ``[x, y]`` argument and return it rounded."""
    if len(values) != 2:
        raise PromptError(f"{name} must have two values: [x, y].")
    _check_unit_range(name, values)
    return _normalize(values[0]), _normalize(values[1])


def _normalize(value: float) -> float:
    """Round a coordinate and fold negative zero into zero."""
    return round(float(value), PROMPT_DECIMALS) + 0.0


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
        """Return the mask ``prompt`` selects and its predicted quality score.

        The mask has the shape of the image handed in.
        """
        pixels = HWC3(np.array(image, dtype=np.uint8))
        digest = hashlib.sha256(pixels.tobytes()).hexdigest()
        if digest != self._embedded_digest:
            logger.info("Encoding image for segmentation (%dx%d)", pixels.shape[1], pixels.shape[0])
            self.predictor.set_image(pixels)
            self._embedded_digest = digest
        height, width = pixels.shape[:2]
        inputs = prediction_inputs(prompt, width, height)
        with torch.no_grad():
            masks, scores, _ = self.predictor.predict(
                point_coords=inputs.point_coords,
                point_labels=inputs.point_labels,
                box=inputs.box,
                multimask_output=inputs.multimask,
            )
        chosen = choose_mask(masks, scores, prompt.extent)
        return masks[chosen].astype(bool), float(scores[chosen])


@dataclass(frozen=True)
class PredictionInputs:
    """A region prompt in the pixel form the mask predictor takes."""

    point_coords: np.ndarray | None
    point_labels: np.ndarray | None
    box: np.ndarray | None
    multimask: bool


def prediction_inputs(prompt: RegionPrompt, width: int, height: int) -> PredictionInputs:
    """Convert a prompt to predictor inputs for an image of ``width`` by ``height`` pixels.

    The include point is labelled 1 and each exclude point 0. Only a lone point
    asks for several candidate masks; the predictor's own guidance is that a
    single mask does better once the prompt is unambiguous. On a dense
    architectural photo, a box that picked the best of several candidates
    spilled onto the neighboring building, while the single mask stayed on
    the building the box enclosed.
    """
    box = None
    if prompt.box is not None:
        x0, y0, x1, y1 = prompt.box
        box = np.array([x0 * width, y0 * height, x1 * width, y1 * height])
    points = [(prompt.point, 1)] if prompt.point is not None else []
    points += [(spot, 0) for spot in prompt.exclude]
    point_coords = None
    point_labels = None
    if points:
        point_coords = np.array([[x * width, y * height] for (x, y), _ in points])
        point_labels = np.array([label for _, label in points])
    return PredictionInputs(point_coords, point_labels, box, multimask=prompt.is_lone_point)


def choose_mask(masks: np.ndarray, scores: np.ndarray, extent: Extent) -> int:
    """Index of the candidate mask ``extent`` asks for.

    best takes the highest predicted quality; largest and smallest compare
    areas rather than relying on the order the candidates come back in.
    """
    if extent == "largest":
        return int(np.argmax(masks.reshape(len(masks), -1).sum(axis=1)))
    if extent == "smallest":
        return int(np.argmin(masks.reshape(len(masks), -1).sum(axis=1)))
    return int(np.argmax(scores))


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
