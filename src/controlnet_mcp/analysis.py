"""Orchestrates one analysis request: reference image, cache, model manager, processor.

This module contains no MCP transport logic; the server module translates its
exceptions into tool errors.
"""

import logging
import threading
from dataclasses import dataclass

from controlnet_mcp.cache import AnalysisCache, image_digest
from controlnet_mcp.config import MAX_RESOLUTION, MIN_RESOLUTION, Settings
from controlnet_mcp.images import (
    decode_reference_image,
    image_to_png_bytes,
    png_size_and_measurement,
    resolve_reference_path,
)
from controlnet_mcp.measurements import Measurement
from controlnet_mcp.model_manager import ModelManager, select_device
from controlnet_mcp.processors import PROCESSORS, ProcessorSpec, get_processor
from controlnet_mcp.regions import FULL_IMAGE, CropError, CropRegion
from controlnet_mcp.segmentation import PromptError, RegionPrompt

logger = logging.getLogger(__name__)


class ResolutionError(ValueError):
    """Raised when a requested detection resolution is outside the accepted range."""


@dataclass(frozen=True)
class AnalysisResult:
    """One generated analysis image plus the facts a tool response reports about it.

    ``measurement`` holds the form of the output's measurement that the
    settings select, and is empty when measurements are switched off. ``crop``
    is the part of the reference that was analyzed, aligned to its pixels, or
    None when the whole image was.
    """

    kind: str
    resolution: int
    png: bytes
    from_cache: bool
    width: int
    height: int
    description: str
    measurement: str
    crop: CropRegion | None = None


@dataclass(frozen=True)
class _CachedRender:
    """The facts read back from a cached PNG that is still fit to serve."""

    width: int
    height: int
    measurement: Measurement


class AnalysisService:
    """Runs analyses over reference images with disk caching and lazy model loading.

    Inference is serialized: the MCP SDK runs tool calls on worker threads, the
    detectors hold mutable state during a run (the segmenter keeps one image
    embedding), and the resident-model bound only holds when a model cannot be
    evicted while another thread is still using it.
    """

    def __init__(
        self, settings: Settings, model_manager: ModelManager, cache: AnalysisCache
    ) -> None:
        self.settings = settings
        self.model_manager = model_manager
        self.cache = cache
        self._inference_lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: Settings) -> "AnalysisService":
        """Build a service whose device, model directory, and cache come from ``settings``."""
        device = select_device(settings.device)
        manager = ModelManager(settings.model_dir, device, settings.max_loaded_models)
        return cls(settings, manager, AnalysisCache(settings.output_dir))

    def analyze(
        self,
        filename: str,
        kind: str,
        resolution: int | None = None,
        prompt: RegionPrompt | None = None,
        crop: CropRegion | None = None,
    ) -> AnalysisResult:
        """Produce the ``kind`` analysis of a reference image at ``resolution``.

        Args:
            filename: Bare file name inside the reference image directory.
            kind: One of the semantic analysis names in the processor registry.
            resolution: Detection resolution; ``None`` uses the configured default.
            prompt: Region to segment; required by analyses that accept a prompt
                and rejected by the others.
            crop: Part of the image to analyze instead of all of it; rejected
                by analyses that accept a prompt.

        Raises:
            ReferenceImageError: When the filename is not an allowed reference image.
            UnknownAnalysisError: When ``kind`` is not registered.
            ResolutionError: When ``resolution`` is outside the accepted range.
            PromptError: When the prompt is missing or not applicable to ``kind``.
            CropError: When the crop is too small or not applicable to ``kind``.
            MissingCheckpointError: When the processor's checkpoint is not installed.
        """
        spec = get_processor(kind)
        resolution = self._validated_resolution(resolution)
        _check_prompt(spec, prompt)
        _check_crop(spec, crop)
        path = resolve_reference_path(self.settings.reference_image_dir, filename)
        data = path.read_bytes()
        digest = image_digest(data)
        image = None
        snapped = None
        variant = prompt.digest() if prompt is not None else None
        if crop is not None:
            image = decode_reference_image(data, filename)
            snapped = crop.snapped(image.width, image.height)
            variant = snapped.digest()

        cached = self.cache.get(digest, spec.kind, spec.version, resolution, variant)
        usable = _usable_cached_render(cached) if cached is not None else None
        if cached is not None and usable is not None:
            logger.info("Serving cached %s for %s at %d", spec.kind, filename, resolution)
            return AnalysisResult(
                kind=spec.kind,
                resolution=resolution,
                png=cached,
                from_cache=True,
                width=usable.width,
                height=usable.height,
                description=(
                    f"{spec.description} Detection resolution {resolution}. Served from cache."
                ),
                measurement=self._selected_form(usable.measurement),
                crop=snapped,
            )
        if cached is not None:
            logger.warning(
                "Cached %s for %s at %d is unreadable or predates measurements; rendering again",
                spec.kind,
                filename,
                resolution,
            )

        region = FULL_IMAGE
        if image is None:
            image = decode_reference_image(data, filename)
        if snapped is not None:
            image = image.crop(snapped.pixel_box(image.width, image.height))
            region = snapped
        with self._inference_lock:
            detector = self.model_manager.get(spec)
            logger.info("Running %s on %s at %d", spec.kind, filename, resolution)
            output = spec.run(detector, image, resolution, prompt, region)
        png = image_to_png_bytes(output.image, output.measurement)
        self.cache.put(digest, spec.kind, spec.version, resolution, png, variant)
        return AnalysisResult(
            kind=spec.kind,
            resolution=resolution,
            png=png,
            from_cache=False,
            width=output.image.width,
            height=output.image.height,
            description=f"{spec.description} Detection resolution {resolution}.",
            measurement=self._selected_form(output.measurement),
            crop=snapped,
        )

    def _selected_form(self, measurement: Measurement) -> str:
        """The measurement text the configured verbosity emits.

        Both forms are always computed and cached, so changing the setting
        changes only what is reported and never invalidates a cached result.
        """
        mode = self.settings.result_measurements
        if mode == "brief":
            return measurement.brief
        if mode == "full":
            return measurement.full
        return ""

    def _validated_resolution(self, resolution: int | None) -> int:
        if resolution is None:
            return self.settings.default_detect_resolution
        if not MIN_RESOLUTION <= resolution <= MAX_RESOLUTION:
            raise ResolutionError(
                f"Resolution {resolution} is outside the accepted range "
                f"{MIN_RESOLUTION} to {MAX_RESOLUTION}."
            )
        return resolution


def _check_prompt(spec: ProcessorSpec, prompt: RegionPrompt | None) -> None:
    """Require a prompt for analyses that take one and reject it everywhere else."""
    if spec.accepts_prompt and prompt is None:
        raise PromptError(
            f"The {spec.kind} analysis needs a box [x0, y0, x1, y1] or a point [x, y] "
            "saying where to look."
        )
    if not spec.accepts_prompt and prompt is not None:
        prompted = ", ".join(kind for kind, entry in PROCESSORS.items() if entry.accepts_prompt)
        raise PromptError(
            f"The {spec.kind} analysis covers the whole image; box and point apply only to "
            f"{prompted}."
        )


def _check_crop(spec: ProcessorSpec, crop: CropRegion | None) -> None:
    """Reject a crop for analyses that take a region prompt, which already choose their region."""
    if crop is not None and spec.accepts_prompt:
        raise CropError(
            f"The {spec.kind} analysis takes no crop; give a box or point to choose its region."
        )


def _usable_cached_render(png: bytes) -> _CachedRender | None:
    """What a cache hit can serve, or None when the file must be rendered again.

    A file that does not decode is unusable. So is one carrying no brief
    measurement: every render stores one, even when there was nothing to find,
    so an empty brief means the file was written before analyses were measured.
    """
    try:
        (width, height), measurement = png_size_and_measurement(png)
    except OSError:
        return None
    if not measurement.brief:
        return None
    return _CachedRender(width=width, height=height, measurement=measurement)
