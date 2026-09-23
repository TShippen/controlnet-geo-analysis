"""Lazy detector loading with a bounded resident set.

Detectors are built on first use from checkpoints already on disk and kept in
a least-recently-used cache whose size is capped by ``MAX_LOADED_MODELS``, so
a long-running server never holds more than the configured number of models in
memory. Nothing here downloads: a missing checkpoint raises
:class:`MissingCheckpointError` naming the files and the preparation command.
"""

from __future__ import annotations

import gc
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from controlnet_mcp.checkpoints import PREPARE_COMMAND, CheckpointSpec, missing_checkpoints

if TYPE_CHECKING:
    from controlnet_mcp.processors import ProcessorSpec

logger = logging.getLogger(__name__)


def select_device(setting: str) -> torch.device:
    """Resolve a configured device setting to a concrete torch device.

    Args:
        setting: One of ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``. ``"auto"``
            prefers CUDA, then MPS, then CPU.

    Returns:
        The device detectors should be built on.

    Raises:
        ValueError: When an explicitly requested backend is unavailable, or when
            the setting names no known backend.
    """
    if setting == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if setting == "cpu":
        return torch.device("cpu")
    if setting == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Device 'cuda' was requested but no CUDA device is available.")
        return torch.device("cuda")
    if setting == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("Device 'mps' was requested but the MPS backend is not available.")
        return torch.device("mps")
    raise ValueError(f"Unknown device setting {setting!r}; expected auto, cpu, cuda, or mps.")


class MissingCheckpointError(Exception):
    """Raised when a processor's checkpoints are absent from the model directory.

    Attributes:
        missing: The checkpoint specs whose files are absent or empty.
    """

    def __init__(self, missing: list[CheckpointSpec]) -> None:
        self.missing = list(missing)
        files = ", ".join(str(spec.relative_path) for spec in self.missing)
        super().__init__(
            f"Missing checkpoint files under the model directory: {files}. "
            f"Run `{PREPARE_COMMAND}` to install them."
        )


class ModelManager:
    """Builds detectors on demand and keeps at most ``max_loaded`` of them resident.

    Processors that need no checkpoints are cheap to construct and are built on
    every call rather than cached. All cache mutation is guarded by a reentrant
    lock, because MCP tool calls run on worker threads.
    """

    def __init__(self, model_dir: Path, device: torch.device, max_loaded: int) -> None:
        """Initialize the manager.

        Args:
            model_dir: Root of the checkpoint layout.
            device: Device passed to every build function.
            max_loaded: Maximum number of resident detectors; must be at least 1.

        Raises:
            ValueError: When ``max_loaded`` is below 1.
        """
        if max_loaded < 1:
            raise ValueError(f"max_loaded must be at least 1, got {max_loaded}.")
        self._model_dir = model_dir
        self._device = device
        self._max_loaded = max_loaded
        self._loaded: OrderedDict[str, object] = OrderedDict()
        self._lock = threading.RLock()

    @property
    def model_dir(self) -> Path:
        """Root directory the checkpoints are loaded from."""
        return self._model_dir

    @property
    def loaded_kinds(self) -> list[str]:
        """Resident analysis kinds, least recently used first."""
        with self._lock:
            return list(self._loaded)

    def get(self, spec: ProcessorSpec) -> object:
        """Return the detector for ``spec``, building and caching it if needed.

        Args:
            spec: The processor whose detector is wanted.

        Returns:
            The detector object produced by ``spec.build``.

        Raises:
            MissingCheckpointError: When any checkpoint the spec needs is absent.
        """
        if not spec.requires_model:
            return spec.build(self._model_dir, self._device)
        with self._lock:
            cached = self._loaded.get(spec.kind)
            if cached is not None:
                self._loaded.move_to_end(spec.kind)
                return cached
            missing = missing_checkpoints(self._model_dir, spec.checkpoints)
            if missing:
                raise MissingCheckpointError(missing)
            while len(self._loaded) >= self._max_loaded:
                evicted_kind, evicted_model = self._loaded.popitem(last=False)
                self._release(evicted_kind, evicted_model)
            model = spec.build(self._model_dir, self._device)
            self._loaded[spec.kind] = model
            logger.info("Loaded %s detector on %s", spec.kind, self._device)
            return model

    def unload(self, kind: str) -> None:
        """Release the detector cached for ``kind``, doing nothing when none is."""
        with self._lock:
            model = self._loaded.pop(kind, None)
            if model is None:
                return
            self._release(kind, model)

    def unload_all(self) -> None:
        """Release every resident detector."""
        with self._lock:
            for kind in list(self._loaded):
                self.unload(kind)

    def _release(self, kind: str, model: object) -> None:
        """Drop the last reference to a detector and reclaim accelerator memory."""
        del model
        gc.collect()
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        elif self._device.type == "mps":
            torch.mps.empty_cache()
        logger.info("Unloaded %s detector from %s", kind, self._device)
