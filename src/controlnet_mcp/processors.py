"""Registry of the six geometric analyses and how to build and run each detector.

Each analysis is described once here: the agent-facing sentence returned with
the image, the checkpoints it needs on disk, how to construct the
``controlnet_aux`` detector for a device, and how to run it at a requested
resolution. The model manager and the analysis service read this registry
rather than knowing anything about ``controlnet_aux`` themselves.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from controlnet_aux import (
    CannyDetector,
    LineartDetector,
    MLSDdetector,
    NormalBaeDetector,
    ZoeDetector,
)
from controlnet_aux.mlsd.utils import pred_lines
from controlnet_aux.segment_anything.build_sam import sam_model_registry
from PIL import Image

from controlnet_mcp.checkpoints import (
    LINEART_CHECKPOINT,
    LINEART_COARSE_CHECKPOINT,
    MLSD_CHECKPOINT,
    MOBILE_SAM_CHECKPOINT,
    NORMALBAE_CHECKPOINT,
    ZOE_CHECKPOINT,
    CheckpointSpec,
    checkpoint_path,
)
from controlnet_mcp.measurements import (
    EMPTY_MEASUREMENT,
    Measurement,
    measure_depth,
    measure_edges,
    measure_lines,
    measure_mask,
    measure_normals,
)
from controlnet_mcp.regions import CropRegion
from controlnet_mcp.segmentation import (
    PromptedSegmenter,
    PromptError,
    RegionPrompt,
    render_region_overlay,
    resize_for_detection,
)

logger = logging.getLogger(__name__)

AnalysisKind = Literal["depth", "normals", "lineart", "lines", "segments", "canny"]

ANALYSIS_KINDS: tuple[str, ...] = ("depth", "normals", "lineart", "lines", "segments", "canny")

ANNOTATOR_SUBDIR = "annotators"

LINE_SCORE_THRESHOLD = 0.1
LINE_DISTANCE_THRESHOLD = 0.1
# Chosen on a dense architectural photo: 6% of the longer side keeps every massing
# outline and all three families of parallel edges while dropping the short
# fragments; 10% already broke building corners and outlines apart.
LONG_LINE_FRACTION = 0.06

LineLength = Literal["all", "long"]


class UnknownAnalysisError(Exception):
    """Raised when a requested analysis name is not one of ``ANALYSIS_KINDS``."""


class OptionError(ValueError):
    """Raised when a per-call option is given to an analysis that does not take it."""


@dataclass(frozen=True)
class AnalysisOptions:
    """Per-call choices that change what one analysis renders.

    A field left at None takes the analysis's default, and only an analysis
    that declares the option accepts a value for it.
    """

    line_length: LineLength | None = None

    def cache_variant(self) -> str | None:
        """The cache key component for these options, or None when all are defaults."""
        if self.line_length is None or self.line_length == "all":
            return None
        return f"length-{self.line_length}"


DEFAULT_OPTIONS = AnalysisOptions()


@dataclass(frozen=True)
class AnalysisOutput:
    """The rendered analysis image and what was measured from it."""

    image: Image.Image
    measurement: Measurement = EMPTY_MEASUREMENT


@dataclass(frozen=True)
class ProcessorSpec:
    """Everything the runtime needs to produce one analysis.

    Attributes:
        kind: The semantic analysis name, one of ``ANALYSIS_KINDS``.
        description: What the image shows and how to read it. Part of the tool
            description, and repeated in the result text when measurements are
            off. Never names a model.
        use_when: Which modeling step the analysis serves and when to skip
            it. Part of the tool description only, so results stay short.
        checkpoints: The checkpoint files the detector loads, empty when the
            detector is purely algorithmic.
        build: Constructs the detector from the model directory and moves it
            to the given device.
        run: Runs a detector on an RGB image at a detect resolution with an
            optional region prompt, the crop region the image came from, and
            the per-call options, and returns the RGB result plus its
            measurement. The crop region lets measured positions be reported
            in fractions of the full reference. Every run measures what it
            rendered, even when there was nothing to find; a cached result
            whose brief measurement is empty is taken for a stale entry and
            rendered again.
        accepts_prompt: Whether the analysis needs a region prompt. Prompts are
            rejected for analyses that do not accept them.
        accepts_line_length: Whether the analysis takes the ``line_length``
            option. The option is rejected for analyses that do not.
        version: Render version, part of every cache key. Bump it whenever the
            output for the same inputs changes: a different checkpoint, a
            changed detector default, or a change to how the result is drawn.
            The stored measurement is part of that output, because a cache hit
            serves the text saved with the PNG rather than measuring again, so
            a change to what the measurement reports needs a bump too.
    """

    kind: str
    description: str
    use_when: str
    checkpoints: tuple[CheckpointSpec, ...]
    build: Callable[[Path, torch.device], object]
    run: Callable[
        [object, Image.Image, int, RegionPrompt | None, CropRegion, AnalysisOptions],
        AnalysisOutput,
    ]
    accepts_prompt: bool = False
    accepts_line_length: bool = False
    version: str = "1"

    @property
    def requires_model(self) -> bool:
        """Whether this analysis loads checkpoints, and so belongs in the model cache."""
        return bool(self.checkpoints)


def _annotators_dir(model_dir: Path) -> str:
    """Directory the ``controlnet_aux`` loaders join their default filenames onto."""
    return str(model_dir / ANNOTATOR_SUBDIR)


def _build_depth(model_dir: Path, device: torch.device) -> object:
    """Build the Zoe relative depth estimator from ``ZoeD_M12_N.pt``."""
    logger.info("Building Zoe depth detector on %s", device)
    return ZoeDetector.from_pretrained(_annotators_dir(model_dir)).to(device)


def _build_normals(model_dir: Path, device: torch.device) -> object:
    """Build the NormalBae surface normal estimator from ``scannet.pt``."""
    logger.info("Building NormalBae detector on %s", device)
    return NormalBaeDetector.from_pretrained(_annotators_dir(model_dir)).to(device)


def _build_lineart(model_dir: Path, device: torch.device) -> object:
    """Build the lineart generator from ``sk_model.pth`` and ``sk_model2.pth``."""
    logger.info("Building lineart detector on %s", device)
    return LineartDetector.from_pretrained(_annotators_dir(model_dir)).to(device)


def _build_lines(model_dir: Path, device: torch.device) -> object:
    """Build the MLSD straight line detector from ``mlsd_large_512_fp32.pth``."""
    logger.info("Building MLSD line detector on %s", device)
    return MLSDdetector.from_pretrained(_annotators_dir(model_dir)).to(device)


def _build_segments(model_dir: Path, device: torch.device) -> object:
    """Build the prompted MobileSAM segmenter from ``mobile_sam.pt``.

    ``SamDetector.from_pretrained`` is not used because it moves the model to
    CUDA whenever CUDA is available and wraps the automatic mask generator,
    whose dense point grid is far slower than prompted prediction.
    """
    logger.info("Building MobileSAM segmenter on %s", device)
    weights = checkpoint_path(model_dir, MOBILE_SAM_CHECKPOINT)
    sam = sam_model_registry["vit_t"](checkpoint=str(weights))
    return PromptedSegmenter(sam, device)


def _build_canny(model_dir: Path, device: torch.device) -> object:
    """Build the Canny edge detector, which reads no checkpoint and ignores the device."""
    del model_dir, device
    return CannyDetector()


def _render_whole_image(detector: object, image: Image.Image, resolution: int) -> Image.Image:
    """Run a whole-image ``controlnet_aux`` detector and return its RGB rendering.

    Every such detector accepts the same three keyword arguments and scales the
    short side of the image to ``resolution``, rounding both sides to multiples
    of 64. Detector-specific thresholds keep their defaults. Whole-image runs
    never see a prompt; the analysis service rejects prompts before this point.
    """
    if not callable(detector):
        raise TypeError(f"Detector {type(detector).__name__} is not callable")
    result = detector(
        image,
        detect_resolution=resolution,
        image_resolution=resolution,
        output_type="pil",
    )
    return result.convert("RGB")


def _run_depth(
    detector: object,
    image: Image.Image,
    resolution: int,
    prompt: RegionPrompt | None,
    region: CropRegion,
    options: AnalysisOptions,
) -> AnalysisOutput:
    """Run the depth detector and measure how much of the map is near, mid, and far."""
    del prompt, options
    rendered = _render_whole_image(detector, image, resolution)
    return AnalysisOutput(
        image=rendered, measurement=measure_depth(_grayscale(rendered), region)
    )


def _run_normals(
    detector: object,
    image: Image.Image,
    resolution: int,
    prompt: RegionPrompt | None,
    region: CropRegion,
    options: AnalysisOptions,
) -> AnalysisOutput:
    """Run the normal detector and measure the flat and curved areas of the map.

    The measurement reports only shares of the map, so the region is unused.
    """
    del prompt, region, options
    rendered = _render_whole_image(detector, image, resolution)
    pixels = np.array(rendered, dtype=np.uint8)
    return AnalysisOutput(image=rendered, measurement=measure_normals(pixels))


def _run_lineart(
    detector: object,
    image: Image.Image,
    resolution: int,
    prompt: RegionPrompt | None,
    region: CropRegion,
    options: AnalysisOptions,
) -> AnalysisOutput:
    """Run the lineart detector and measure its edge density; its lines are light on dark."""
    del prompt, region, options
    rendered = _render_whole_image(detector, image, resolution)
    measurement = measure_edges(_grayscale(rendered), edges_are_dark=False)
    return AnalysisOutput(image=rendered, measurement=measurement)


def _run_canny(
    detector: object,
    image: Image.Image,
    resolution: int,
    prompt: RegionPrompt | None,
    region: CropRegion,
    options: AnalysisOptions,
) -> AnalysisOutput:
    """Run the Canny detector and measure its edge density; its edges are white on black."""
    del prompt, region, options
    rendered = _render_whole_image(detector, image, resolution)
    measurement = measure_edges(_grayscale(rendered), edges_are_dark=False)
    return AnalysisOutput(image=rendered, measurement=measurement)


def _run_lines(
    detector: object,
    image: Image.Image,
    resolution: int,
    prompt: RegionPrompt | None,
    region: CropRegion,
    options: AnalysisOptions,
) -> AnalysisOutput:
    """Draw the detected straight segments and measure their endpoints.

    The drawing here mirrors what the detector does in its own call, one-pixel
    white segments on a black canvas at the detect resolution, rather than
    calling it: running the prediction directly keeps the endpoints, which the
    detector would discard after drawing them. With ``line_length`` set to
    long, only segments at least ``LONG_LINE_FRACTION`` of the longer side are
    drawn and measured.
    """
    del prompt
    if not isinstance(detector, MLSDdetector):
        raise TypeError(f"Expected an MLSDdetector, got {type(detector).__name__}")
    pixels = np.array(resize_for_detection(image, resolution), dtype=np.uint8)
    height, width = pixels.shape[:2]
    segments = _predict_line_segments(detector, pixels)
    long_only = options.line_length == "long"
    if long_only:
        segments = _long_segments(segments, LONG_LINE_FRACTION * max(width, height))
    canvas = np.zeros_like(pixels)
    for x0, y0, x1, y1 in segments:
        cv2.line(canvas, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 255), 1)
    logger.info("Kept %d straight segments at %dx%d", len(segments), width, height)
    return AnalysisOutput(
        image=Image.fromarray(canvas),
        measurement=measure_lines(segments.tolist(), width, height, region, long_only),
    )


def _long_segments(segments: np.ndarray, min_length: float) -> np.ndarray:
    """The segments whose length in pixels is at least ``min_length``."""
    lengths = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
    return segments[lengths >= min_length]


def _predict_line_segments(detector: MLSDdetector, pixels: np.ndarray) -> np.ndarray:
    """Endpoint quadruples of every straight segment found in an RGB pixel array.

    ``pred_lines`` scales the columns of its result without first checking that
    it found anything, so an image with no straight edges raises ``IndexError``
    there instead of returning an empty array. That case is a finding of none,
    not a failure, so it is reported as an empty result.
    """
    with torch.no_grad():
        try:
            return pred_lines(
                pixels,
                detector.model,
                [pixels.shape[0], pixels.shape[1]],
                LINE_SCORE_THRESHOLD,
                LINE_DISTANCE_THRESHOLD,
            )
        except IndexError:
            logger.info("No straight segments found")
            return np.zeros((0, 4), dtype=np.float32)


def _grayscale(image: Image.Image) -> np.ndarray:
    """The 8-bit luminance channel of a rendered analysis."""
    return np.array(image.convert("L"), dtype=np.uint8)


def _run_segments(
    detector: object,
    image: Image.Image,
    resolution: int,
    prompt: RegionPrompt | None,
    region: CropRegion,
    options: AnalysisOptions,
) -> AnalysisOutput:
    """Segment the prompted region, render it as an overlay, and measure the mask.

    Segmentation always sees the whole reference, since the analysis service
    rejects a crop for it, so the region is unused.
    """
    del region, options
    if prompt is None:
        raise PromptError("Segmentation needs a box [x0, y0, x1, y1] or a point [x, y].")
    if not isinstance(detector, PromptedSegmenter):
        raise TypeError(f"Expected a PromptedSegmenter, got {type(detector).__name__}")
    resized = resize_for_detection(image, resolution)
    mask, score = detector.segment(resized, prompt)
    logger.info(
        "Segmented region with score %.3f covering %.1f%% of the image", score, 100 * mask.mean()
    )
    return AnalysisOutput(
        image=render_region_overlay(resized, mask), measurement=measure_mask(mask)
    )


PROCESSORS: dict[str, ProcessorSpec] = {
    "depth": ProcessorSpec(
        kind="depth",
        description=(
            "Relative depth map: brighter is closer. Gray levels are stretched for each image, "
            "so they show which surfaces are in front and roughly how far apart they sit within "
            "this image, never distances or ratios. The farthest part of every scene is solid "
            "black."
        ),
        use_when=(
            "Use it to order parts front to back and to see which faces step back. Do not take "
            "sizes or extrusion lengths from it; get those from straight edges and one known "
            "dimension."
        ),
        checkpoints=(ZOE_CHECKPOINT,),
        build=_build_depth,
        run=_run_depth,
        version="2",
    ),
    "normals": ProcessorSpec(
        kind="normals",
        description=(
            "Surface direction map as seen from the camera. More blue faces the viewer, more red "
            "faces left and less red faces right, more green faces up, so the same face changes "
            "color when the view changes. A flat face is one even color, a curved surface shades "
            "smoothly, a crease is a sharp color change, and a fillet is a narrow gradient."
        ),
        use_when=(
            "Use it to choose a surface type for each part: an even color means a plane (planar "
            "surface, extrusion, box) and smooth shading means a curve (revolve, loft, sweep, "
            "network surface, SubD). It also tells a crease from a fillet. It is least reliable "
            "on large plain surfaces such as floors."
        ),
        checkpoints=(NORMALBAE_CHECKPOINT,),
        build=_build_normals,
        run=_run_normals,
    ),
    "lineart": ProcessorSpec(
        kind="lineart",
        description=(
            "Line drawing of the edges that carry shape: silhouettes, creases, and part "
            "boundaries, drawn light on black with texture and shading removed. It does not say "
            "which kind of edge a line is."
        ),
        use_when=(
            "Use it to trace outlines and profiles into curves and to find where one part ends "
            "and the next begins. It can drop faint details, so check the original before "
            "relying on a missing edge."
        ),
        checkpoints=(LINEART_CHECKPOINT, LINEART_COARSE_CHECKPOINT),
        build=_build_lineart,
        run=_run_lineart,
        version="2",
    ),
    "lines": ProcessorSpec(
        kind="lines",
        description=(
            "Straight edges only, drawn as white segments on black. Curves are missed or broken "
            "into short pieces."
        ),
        use_when=(
            "Use it first on objects built from straight parts: to find the main axes and the "
            "direction of perspective, to compare proportions, and to trace straight edges. Set "
            "line_length to long to keep only the main edges when finding axes. Skip it for "
            "organic or mostly curved objects."
        ),
        checkpoints=(MLSD_CHECKPOINT,),
        build=_build_lines,
        run=_run_lines,
        accepts_line_length=True,
        version="2",
    ),
    "segments": ProcessorSpec(
        kind="segments",
        description=(
            "One region, chosen with a box or point, tinted and outlined on the photo. The "
            "outline is flat: it says nothing about depth or about parts hidden from view. A "
            "single point can select the whole object, one part, or a smaller piece of a part."
        ),
        use_when=(
            "Use it to split the object into components and to mark the region you will study "
            "in depth or normals. Give a box when you mean a whole component."
        ),
        checkpoints=(MOBILE_SAM_CHECKPOINT,),
        build=_build_segments,
        run=_run_segments,
        accepts_prompt=True,
    ),
    "canny": ProcessorSpec(
        kind="canny",
        description=(
            "Every sharp change in brightness or color, white on black: real edges mixed with "
            "texture, shadows, highlights, and noise."
        ),
        use_when=(
            "Use it only to confirm a detail the line drawing dropped, in an area you already "
            "know is solid geometry. Don't start with it, and don't trace outlines from it."
        ),
        checkpoints=(),
        build=_build_canny,
        run=_run_canny,
    ),
}


def get_processor(kind: str) -> ProcessorSpec:
    """Look up the processor for an analysis name.

    Raises:
        UnknownAnalysisError: When ``kind`` is not a registered analysis.
    """
    try:
        return PROCESSORS[kind]
    except KeyError:
        raise UnknownAnalysisError(
            f"Unknown analysis {kind!r}; supported analyses: " + ", ".join(ANALYSIS_KINDS)
        ) from None
