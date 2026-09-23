"""Orchestrates one analysis request: reference image, cache, model manager, processor.

This module contains no MCP transport logic; the server module translates its
exceptions into tool errors.
"""

import io
import logging
import threading
from dataclasses import dataclass

from PIL import Image

from controlnet_mcp.cache import AnalysisCache, image_digest
from controlnet_mcp.config import MAX_RESOLUTION, MIN_RESOLUTION, Settings
from controlnet_mcp.images import (
    decode_reference_image,
    image_to_png_bytes,
    png_note,
    resolve_reference_path,
)
from controlnet_mcp.model_manager import ModelManager, select_device
from controlnet_mcp.processors import PROCESSORS, ProcessorSpec, get_processor
from controlnet_mcp.segmentation import PromptError, RegionPrompt

logger = logging.getLogger(__name__)


class ResolutionError(ValueError):
    """Raised when a requested detection resolution is outside the accepted range."""


@dataclass(frozen=True)
class AnalysisResult:
    """One generated analysis image plus the facts a tool response reports about it."""

    kind: str
    resolution: int
    png: bytes
    from_cache: bool
    width: int
    height: int
    description: str
    note: str


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
    ) -> AnalysisResult:
        """Produce the ``kind`` analysis of a reference image at ``resolution``.

        Args:
            filename: Bare file name inside the reference image directory.
            kind: One of the semantic analysis names in the processor registry.
            resolution: Detection resolution; ``None`` uses the configured default.
            prompt: Region to segment; required by analyses that accept a prompt
                and rejected by the others.

        Raises:
            ReferenceImageError: When the filename is not an allowed reference image.
            UnknownAnalysisError: When ``kind`` is not registered.
            ResolutionError: When ``resolution`` is outside the accepted range.
            PromptError: When the prompt is missing or not applicable to ``kind``.
            MissingCheckpointError: When the processor's checkpoint is not installed.
        """
        spec = get_processor(kind)
        resolution = self._validated_resolution(resolution)
        _check_prompt(spec, prompt)
        variant = prompt.digest() if prompt is not None else None
        path = resolve_reference_path(self.settings.reference_image_dir, filename)
        data = path.read_bytes()
        digest = image_digest(data)

        cached = self.cache.get(digest, spec.kind, spec.version, resolution, variant)
        cached_size = _png_size(cached) if cached is not None else None
        if cached is not None and cached_size is not None:
            width, height = cached_size
            logger.info("Serving cached %s for %s at %d", spec.kind, filename, resolution)
            return AnalysisResult(
                kind=spec.kind,
                resolution=resolution,
                png=cached,
                from_cache=True,
                width=width,
                height=height,
                description=(
                    f"{spec.description} Detection resolution {resolution}. Served from cache."
                ),
                note=png_note(cached),
            )
        if cached is not None:
            logger.warning(
                "Cached %s for %s at %d is unreadable; rendering again",
                spec.kind,
                filename,
                resolution,
            )

        image = decode_reference_image(data, filename)
        with self._inference_lock:
            detector = self.model_manager.get(spec)
            logger.info("Running %s on %s at %d", spec.kind, filename, resolution)
            output = spec.run(detector, image, resolution, prompt)
        png = image_to_png_bytes(output.image, output.note)
        self.cache.put(digest, spec.kind, spec.version, resolution, png, variant)
        return AnalysisResult(
            kind=spec.kind,
            resolution=resolution,
            png=png,
            from_cache=False,
            width=output.image.width,
            height=output.image.height,
            description=f"{spec.description} Detection resolution {resolution}.",
            note=output.note,
        )

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


def _png_size(png: bytes) -> tuple[int, int] | None:
    """Width and height of a cached PNG, or None when the bytes do not decode."""
    try:
        with Image.open(io.BytesIO(png)) as image:
            image.load()
            return image.width, image.height
    except OSError:
        return None
