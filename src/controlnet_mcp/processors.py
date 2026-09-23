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

import torch
from controlnet_aux import (
    CannyDetector,
    LineartDetector,
    MLSDdetector,
    NormalBaeDetector,
    ZoeDetector,
)
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
from controlnet_mcp.segmentation import (
    PromptedSegmenter,
    PromptError,
    RegionPrompt,
    describe_region,
    render_region_overlay,
    resize_for_detection,
)

logger = logging.getLogger(__name__)

AnalysisKind = Literal["depth", "normals", "lineart", "lines", "segments", "canny"]

ANALYSIS_KINDS: tuple[str, ...] = ("depth", "normals", "lineart", "lines", "segments", "canny")

ANNOTATOR_SUBDIR = "annotators"


class UnknownAnalysisError(Exception):
    """Raised when a requested analysis name is not one of ``ANALYSIS_KINDS``."""


@dataclass(frozen=True)
class AnalysisOutput:
    """The rendered analysis image and an optional note describing what was found."""

    image: Image.Image
    note: str = ""


@dataclass(frozen=True)
class ProcessorSpec:
    """Everything the runtime needs to produce one analysis.

    Attributes:
        kind: The semantic analysis name, one of ``ANALYSIS_KINDS``.
        description: One sentence telling an agent what the image shows and
            when to ask for it; returned alongside the image. Never names a model.
        checkpoints: The checkpoint files the detector loads, empty when the
            detector is purely algorithmic.
        build: Constructs the detector from the model directory and moves it
            to the given device.
        run: Runs a detector on an RGB image at a detect resolution with an
            optional region prompt and returns the RGB result plus a note.
        accepts_prompt: Whether the analysis needs a region prompt. Prompts are
            rejected for analyses that do not accept them.
    """

    kind: str
    description: str
    checkpoints: tuple[CheckpointSpec, ...]
    build: Callable[[Path, torch.device], object]
    run: Callable[[object, Image.Image, int, RegionPrompt | None], AnalysisOutput]
    accepts_prompt: bool = False

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


def _run_detector(
    detector: object, image: Image.Image, resolution: int, prompt: RegionPrompt | None
) -> AnalysisOutput:
    """Run a whole-image ``controlnet_aux`` detector and return an RGB result.

    Every such detector accepts the same three keyword arguments and scales the
    short side of the image to ``resolution``, rounding both sides to multiples
    of 64. Detector-specific thresholds keep their defaults. ``prompt`` is
    always None here; the analysis service rejects prompts before this point.
    """
    del prompt
    if not callable(detector):
        raise TypeError(f"Detector {type(detector).__name__} is not callable")
    result = detector(
        image,
        detect_resolution=resolution,
        image_resolution=resolution,
        output_type="pil",
    )
    return AnalysisOutput(image=result.convert("RGB"))


def _run_segments(
    detector: object, image: Image.Image, resolution: int, prompt: RegionPrompt | None
) -> AnalysisOutput:
    """Segment the prompted region and render it as an overlay with a summary note."""
    if prompt is None:
        raise PromptError("Segmentation needs a box [x0, y0, x1, y1] or a point [x, y].")
    if not isinstance(detector, PromptedSegmenter):
        raise TypeError(f"Expected a PromptedSegmenter, got {type(detector).__name__}")
    resized = resize_for_detection(image, resolution)
    mask, score = detector.segment(resized, prompt)
    logger.info(
        "Segmented region with score %.3f covering %.1f%% of the image", score, 100 * mask.mean()
    )
    return AnalysisOutput(image=render_region_overlay(resized, mask), note=describe_region(mask))


PROCESSORS: dict[str, ProcessorSpec] = {
    "depth": ProcessorSpec(
        kind="depth",
        description=(
            "Depth map: brighter pixels are closer to the camera. Ask for it to judge which "
            "parts sit in front of others, where surfaces step back, and how deep to extrude."
        ),
        checkpoints=(ZOE_CHECKPOINT,),
        build=_build_depth,
        run=_run_detector,
    ),
    "normals": ProcessorSpec(
        kind="normals",
        description=(
            "Surface orientation map: each color is a facing direction, so flat faces are one "
            "even color and curved surfaces shade smoothly. Ask for it to tell planes from "
            "curves and to read the tilt of each face."
        ),
        checkpoints=(NORMALBAE_CHECKPOINT,),
        build=_build_normals,
        run=_run_detector,
    ),
    "lineart": ProcessorSpec(
        kind="lineart",
        description=(
            "Clean line drawing: dark contours on white with texture and shading removed. Ask "
            "for it to trace silhouettes, part boundaries, and profile curves."
        ),
        checkpoints=(LINEART_CHECKPOINT, LINEART_COARSE_CHECKPOINT),
        build=_build_lineart,
        run=_run_detector,
    ),
    "lines": ProcessorSpec(
        kind="lines",
        description=(
            "Straight edges only: white segments on black. Ask for it to find principal axes, "
            "planar edges, and perspective direction in objects with straight geometry."
        ),
        checkpoints=(MLSD_CHECKPOINT,),
        build=_build_lines,
        run=_run_detector,
    ),
    "segments": ProcessorSpec(
        kind="segments",
        description=(
            "Region outline: the part you pointed at is tinted and outlined on the image. Ask "
            "for it with a box or point to isolate one component and learn its extent."
        ),
        checkpoints=(MOBILE_SAM_CHECKPOINT,),
        build=_build_segments,
        run=_run_segments,
        accepts_prompt=True,
    ),
    "canny": ProcessorSpec(
        kind="canny",
        description=(
            "Raw edge pixels: every sharp intensity change, including texture and noise. Ask "
            "for it when the clean line drawing dropped a detail you need."
        ),
        checkpoints=(),
        build=_build_canny,
        run=_run_detector,
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
