"""Registry of the geometric analyses and how to build and run each detector.

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
    BEYOND_RANGE_MAX,
    EMPTY_MEASUREMENT,
    GROUP_COLORS,
    Measurement,
    measure_depth,
    measure_edges,
    measure_lines,
    measure_mask,
    measure_normals,
    measure_perspective,
)
from controlnet_mcp.perspective import (
    COLLINEAR_PIXELS,
    INLIER_DEGREES,
    Horizon,
    analyze_perspective,
)
from controlnet_mcp.regions import FULL_IMAGE, CropRegion
from controlnet_mcp.sampling import ValueMap
from controlnet_mcp.segmentation import (
    PromptedSegmenter,
    PromptError,
    RegionPrompt,
    render_region_overlay,
    resize_for_detection,
)

logger = logging.getLogger(__name__)

AnalysisKind = Literal[
    "depth", "normals", "lineart", "lines", "perspective", "segments", "canny"
]

ANALYSIS_KINDS: tuple[str, ...] = (
    "depth",
    "normals",
    "lineart",
    "lines",
    "perspective",
    "segments",
    "canny",
)

SampledKind = Literal["depth", "normals"]

ANNOTATOR_SUBDIR = "annotators"

LINE_SCORE_THRESHOLD = 0.1
LINE_DISTANCE_THRESHOLD = 0.1
# Chosen on a dense architectural photo: 6% of the longer side keeps every massing
# outline and all three families of parallel edges while dropping the short
# fragments; 10% already broke building corners and outlines apart.
LONG_LINE_FRACTION = 0.06

LineLength = Literal["all", "long"]

UNASSIGNED_COLOR = (64, 64, 64)
HORIZON_COLOR = (255, 255, 255)


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
        read_values: Decodes the rendered RGB array into the values it
            encodes, for analyses whose output can be sampled at a position.
            None for the others.
        values_description: What the sampled values of this analysis are and
            are not. Part of the sampling tool's description, so it never
            names a model. Set whenever ``read_values`` is.
        detector: The analysis whose detector this one runs, when it has none
            of its own. Analyses naming the same detector share one loaded
            model. None when the analysis has its own detector.
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
    read_values: Callable[[np.ndarray], ValueMap] | None = None
    values_description: str = ""
    detector: str | None = None

    @property
    def detector_key(self) -> str:
        """The name the detector is loaded and kept under: ``detector``, or else ``kind``."""
        return self.detector or self.kind

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
    """Run the normal detector and measure the flat faces and curved area of the map."""
    del prompt, options
    rendered = _render_whole_image(detector, image, resolution)
    pixels = np.array(rendered, dtype=np.uint8)
    return AnalysisOutput(image=rendered, measurement=measure_normals(pixels, region))


def read_depth_values(rgb: np.ndarray) -> ValueMap:
    """Decode a rendered depth map into levels.

    The level is the gray value, 0 to 255, where higher is closer. A pixel at
    or below ``BEYOND_RANGE_MAX`` lies beyond the depth range and has no level.
    """
    levels = rgb.astype(np.float32).mean(axis=2)
    return ValueMap(
        values=levels[:, :, np.newaxis], beyond_range=levels <= BEYOND_RANGE_MAX, vector=False
    )


def read_normal_values(rgb: np.ndarray) -> ValueMap:
    """Decode a rendered normal map into unit directions in camera axes.

    The components are right, up, and toward the camera. Red is high for a
    face turned to the image left, so the right component is the negated
    decoded red. Every pixel of a normal map carries a direction.
    """
    decoded = rgb.astype(np.float32) / 255.0 * 2.0 - 1.0
    directions = decoded * np.array([-1.0, 1.0, 1.0], dtype=np.float32)
    lengths = np.linalg.norm(directions, axis=2, keepdims=True)
    directions = directions / np.maximum(lengths, np.finfo(np.float32).eps)
    return ValueMap(
        values=directions, beyond_range=np.zeros(rgb.shape[:2], dtype=bool), vector=True
    )


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
    segments, width, height = detect_line_segments(detector, image, resolution)
    long_only = options.line_length == "long"
    if long_only:
        segments = _long_segments(segments, LONG_LINE_FRACTION * max(width, height))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for segment in segments:
        _draw_segment(canvas, segment, (255, 255, 255))
    logger.info("Kept %d straight segments at %dx%d", len(segments), width, height)
    return AnalysisOutput(
        image=Image.fromarray(canvas),
        measurement=measure_lines(segments.tolist(), width, height, region, long_only),
    )


def _run_perspective(
    detector: object,
    image: Image.Image,
    resolution: int,
    prompt: RegionPrompt | None,
    region: CropRegion,
    options: AnalysisOptions,
) -> AnalysisOutput:
    """Group the detected straight segments by perspective, draw the groups, and measure them.

    Each group is drawn in its own color, segments in no group are drawn dark
    gray, and the horizon, when it could be placed, is drawn white across the
    frame beneath the segments. An image analyzed in part is treated as a
    crop, which withholds the camera estimate.
    """
    del prompt, options
    segments, width, height = detect_line_segments(detector, image, resolution)
    result = analyze_perspective(segments, width, height, cropped=region != FULL_IMAGE)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if isinstance(result.horizon, Horizon):
        left = (0, round(result.horizon.left_y * height))
        right = (width - 1, round(result.horizon.right_y * height))
        cv2.line(canvas, left, right, HORIZON_COLOR, 1)
    for segment in segments:
        _draw_segment(canvas, segment, UNASSIGNED_COLOR)
    for family, (_, color) in zip(result.families, GROUP_COLORS, strict=False):
        for index in family.segment_indices:
            _draw_segment(canvas, segments[index], color)
    logger.info(
        "Grouped %d straight segments into %d families at %dx%d",
        len(segments),
        len(result.families),
        width,
        height,
    )
    return AnalysisOutput(
        image=Image.fromarray(canvas),
        measurement=measure_perspective(result, width, height, region),
    )


def detect_line_segments(
    detector: object, image: Image.Image, resolution: int
) -> tuple[np.ndarray, int, int]:
    """Detect the straight segments of an image at a detect resolution.

    Args:
        detector: The detector built for the lines analysis.
        image: The RGB image to detect in.
        resolution: The detect resolution.

    Returns:
        The endpoint quadruples ``x0, y0, x1, y1`` in detection pixels, shaped
        (N, 4), and the width and height of the frame they were detected in.

    Raises:
        TypeError: When ``detector`` is not the line detector.
    """
    if not isinstance(detector, MLSDdetector):
        raise TypeError(f"Expected an MLSDdetector, got {type(detector).__name__}")
    pixels = np.array(resize_for_detection(image, resolution), dtype=np.uint8)
    height, width = pixels.shape[:2]
    return _predict_line_segments(detector, pixels), width, height


def _draw_segment(canvas: np.ndarray, segment: np.ndarray, color: tuple[int, int, int]) -> None:
    """Draw one segment a pixel wide onto an RGB canvas."""
    x0, y0, x1, y1 = segment
    cv2.line(canvas, (int(x0), int(y0)), (int(x1), int(y1)), color, 1)


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
        read_values=read_depth_values,
        values_description=(
            "A level from 0 to 255, higher is closer. Levels order surfaces within one image "
            "and one crop. Differences between levels are not distances, ratios of levels mean "
            "nothing, and levels from another crop, resolution, or image are not comparable. A "
            "sample beyond the depth range has no level. Levels change gradually across a step "
            "between two surfaces, over several pixels, so a single sample there is rarely "
            "flagged as on a boundary; to find a step, sample along a line across it and read "
            "the changes. Sky and open background usually still "
            "get a level, with a small spread, so a steady reading does not show that a surface "
            "is there; check the position against the original image."
        ),
    ),
    "normals": ProcessorSpec(
        kind="normals",
        description=(
            "Surface direction map as seen from the camera. More blue faces the viewer, more red "
            "faces left and less red faces right, more green faces up, so the same face changes "
            "color when the view changes. A flat face is one even color, a curved surface shades "
            "smoothly, a crease is a sharp color change, and a fillet is a narrow gradient. Sky "
            "and open background are colored too and can look like a flat face."
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
        version="3",
        read_values=read_normal_values,
        values_description=(
            "The direction a surface faces, as the components [right, up, toward the camera] "
            "of a unit vector. The direction is relative to this camera view, not to the world: "
            "a tilted camera tilts every value, and the same face reads differently in another "
            "view. Sky and open background still get a direction, with a small spread, so a "
            "steady reading does not show that a surface is there; check the position against "
            "the original image."
        ),
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
            "line_length to long to keep only the main edges when finding axes. A busy scene "
            "can hit the limit on detected edges, so crop to the part you need when edges are "
            "missing. Skip it for organic or mostly curved objects."
        ),
        checkpoints=(MLSD_CHECKPOINT,),
        build=_build_lines,
        run=_run_lines,
        accepts_line_length=True,
        version="2",
    ),
    "perspective": ProcessorSpec(
        kind="perspective",
        description=(
            "Straight edges grouped by the direction they run in the scene. The first group is "
            "drawn red, the second green, the third blue, and edges in no group dark gray. The "
            "edges of a group either meet at one vanishing point, which often lies outside the "
            "image, or run parallel. A white line is the horizon, drawn only when it could be "
            "placed."
        ),
        use_when=(
            "Use it on scenes with straight parallel edges: to find where the main directions "
            "converge, which edges share one line, and, when the evidence allows, the field of "
            "view and tilt of the camera. It reports groups, not meanings. Which group is "
            "vertical in the scene, and whether the groups are perpendicular, are assumptions, "
            "and the result names the ones it made. A crop, fewer than two converging groups, "
            "or groups that contradict being perpendicular withhold the camera estimate, and "
            "the result gives the reason. Curved or organic subjects produce no groups, and "
            "photos with straightened verticals show a parallel group. An edge joins a group "
            f"when it points within {INLIER_DEGREES:.0f} degrees of that group's vanishing "
            "point, so a few edges of a group may belong to another direction. Edges are "
            f"reported as sharing one line when each lies within {COLLINEAR_PIXELS:.0f} pixels "
            "of the other's line, so closely spaced parallel edges can be chained together. A "
            "vanishing point far outside the image is placed less precisely than a near one, "
            "and so is whatever is derived from it. It gives no position, distance, or size."
        ),
        checkpoints=(MLSD_CHECKPOINT,),
        build=_build_lines,
        run=_run_perspective,
        detector="lines",
    ),
    "segments": ProcessorSpec(
        kind="segments",
        description=(
            "One region, chosen with a box or point, tinted and outlined on the photo. The "
            "outline is flat: it says nothing about depth or about parts hidden from view. A "
            "lone point is ambiguous: it can mean the whole object, one part, or a smaller "
            "piece of a part."
        ),
        use_when=(
            "Use it to split the object into components and to mark the region you will study "
            "in depth or normals. Give a box when you mean a whole component. With a lone "
            "point, set extent to largest for the whole object or smallest for the piece under "
            "the point. Add exclude points on neighbors the region should not swallow."
        ),
        checkpoints=(MOBILE_SAM_CHECKPOINT,),
        build=_build_segments,
        run=_run_segments,
        accepts_prompt=True,
        version="2",
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


SAMPLED_KINDS: tuple[str, ...] = tuple(
    kind for kind, spec in PROCESSORS.items() if spec.read_values is not None
)


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
