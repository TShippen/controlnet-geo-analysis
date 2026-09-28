"""Tests for device selection and the bounded detector cache."""

import weakref
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from controlnet_mcp.checkpoints import CheckpointSpec, checkpoint_path
from controlnet_mcp.model_manager import MissingCheckpointError, ModelManager, select_device


@dataclass(frozen=True)
class FakeProcessorSpec:
    """Stand-in carrying only the members ``ModelManager`` reads off a processor spec."""

    kind: str
    checkpoints: tuple[CheckpointSpec, ...]
    build: Callable[[Path, torch.device], object]
    detector: str | None = None

    @property
    def detector_key(self) -> str:
        """The name the detector is cached under."""
        return self.detector or self.kind

    @property
    def requires_model(self) -> bool:
        """Whether this processor needs checkpoints on disk."""
        return bool(self.checkpoints)


class RecordingBuild:
    """Build callable that records its arguments and returns a fresh object each call."""

    def __init__(self) -> None:
        self.calls: list[tuple[Path, torch.device]] = []

    def __call__(self, model_dir: Path, device: torch.device) -> object:
        self.calls.append((model_dir, device))
        return object()


def write_test_checkpoint(model_dir: Path, filename: str) -> CheckpointSpec:
    """Create a non-empty stand-in checkpoint under ``model_dir`` and return its spec."""
    spec = CheckpointSpec("test/repo", filename, Path("annotators") / filename)
    path = checkpoint_path(model_dir, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"w")
    return spec


def make_spec(model_dir: Path, kind: str) -> FakeProcessorSpec:
    """Build a learned-processor stand-in whose one checkpoint exists on disk."""
    return FakeProcessorSpec(
        kind=kind,
        checkpoints=(write_test_checkpoint(model_dir, f"{kind}.pt"),),
        build=RecordingBuild(),
    )


def test_select_device_cpu_explicit() -> None:
    assert select_device("cpu") == torch.device("cpu")


def test_select_device_cuda_unavailable_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError):
        select_device("cuda")


def test_select_device_auto_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert select_device("auto") == torch.device("cpu")


def test_select_device_unknown_setting_raises() -> None:
    with pytest.raises(ValueError):
        select_device("tpu")


def test_max_loaded_below_one_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ModelManager(tmp_path, torch.device("cpu"), max_loaded=0)


def test_get_builds_once_and_reuses(tmp_path: Path) -> None:
    build = RecordingBuild()
    checkpoint = write_test_checkpoint(tmp_path, "depth.pt")
    spec = FakeProcessorSpec(kind="depth", checkpoints=(checkpoint,), build=build)
    manager = ModelManager(tmp_path, torch.device("cpu"), max_loaded=1)

    first = manager.get(spec)
    second = manager.get(spec)

    assert first is second
    assert len(build.calls) == 1
    assert build.calls[0] == (tmp_path, torch.device("cpu"))


def test_specs_sharing_a_detector_load_it_once(tmp_path: Path) -> None:
    build = RecordingBuild()
    checkpoint = write_test_checkpoint(tmp_path, "lines.pt")
    lines = FakeProcessorSpec(kind="lines", checkpoints=(checkpoint,), build=build)
    perspective = FakeProcessorSpec(
        kind="perspective", checkpoints=(checkpoint,), build=build, detector="lines"
    )
    manager = ModelManager(tmp_path, torch.device("cpu"), max_loaded=1)

    first = manager.get(lines)
    second = manager.get(perspective)

    assert first is second
    assert len(build.calls) == 1
    assert manager.loaded_detectors == ["lines"]


def test_get_evicts_least_recently_used(tmp_path: Path) -> None:
    depth = make_spec(tmp_path, "depth")
    normals = make_spec(tmp_path, "normals")
    manager = ModelManager(tmp_path, torch.device("cpu"), max_loaded=1)

    manager.get(depth)
    manager.get(normals)

    assert manager.loaded_detectors == ["normals"]


class FakeModel:
    """Weak-referenceable stand-in for a loaded detector."""


def test_evicted_model_is_freed_before_replacement_builds(tmp_path: Path) -> None:
    depth = FakeProcessorSpec(
        kind="depth",
        checkpoints=(write_test_checkpoint(tmp_path, "depth.pt"),),
        build=lambda model_dir, device: FakeModel(),
    )
    manager = ModelManager(tmp_path, torch.device("cpu"), max_loaded=1)
    depth_ref = weakref.ref(manager.get(depth))
    alive_at_build: list[bool] = []

    def build_normals(model_dir: Path, device: torch.device) -> object:
        alive_at_build.append(depth_ref() is not None)
        return object()

    normals = FakeProcessorSpec(
        kind="normals",
        checkpoints=(write_test_checkpoint(tmp_path, "normals.pt"),),
        build=build_normals,
    )

    manager.get(normals)

    assert alive_at_build == [False]


def test_get_respects_max_loaded_two(tmp_path: Path) -> None:
    depth = make_spec(tmp_path, "depth")
    normals = make_spec(tmp_path, "normals")
    lineart = make_spec(tmp_path, "lineart")
    manager = ModelManager(tmp_path, torch.device("cpu"), max_loaded=2)

    manager.get(depth)
    manager.get(normals)
    manager.get(depth)
    manager.get(lineart)

    assert manager.loaded_detectors == ["depth", "lineart"]


def test_missing_checkpoint_error_carries_the_missing_checkpoint(tmp_path: Path) -> None:
    absent = CheckpointSpec("test/repo", "gone.pt", Path("annotators/gone.pt"))
    build = RecordingBuild()
    spec = FakeProcessorSpec(kind="depth", checkpoints=(absent,), build=build)
    manager = ModelManager(tmp_path, torch.device("cpu"), max_loaded=1)

    with pytest.raises(MissingCheckpointError) as excinfo:
        manager.get(spec)

    assert excinfo.value.missing == [absent]
    assert build.calls == []


def test_modelless_spec_not_cached(tmp_path: Path) -> None:
    build = RecordingBuild()
    spec = FakeProcessorSpec(kind="canny", checkpoints=(), build=build)
    manager = ModelManager(tmp_path, torch.device("cpu"), max_loaded=1)

    first = manager.get(spec)
    second = manager.get(spec)

    assert first is not second
    assert len(build.calls) == 2
    assert manager.loaded_detectors == []


def test_unload_all_clears(tmp_path: Path) -> None:
    depth = make_spec(tmp_path, "depth")
    normals = make_spec(tmp_path, "normals")
    manager = ModelManager(tmp_path, torch.device("cpu"), max_loaded=2)
    manager.get(depth)
    manager.get(normals)

    manager.unload_all()

    assert manager.loaded_detectors == []


def test_unload_missing_kind_is_noop(tmp_path: Path) -> None:
    depth = make_spec(tmp_path, "depth")
    manager = ModelManager(tmp_path, torch.device("cpu"), max_loaded=2)
    manager.get(depth)

    manager.unload("segments")

    assert manager.loaded_detectors == ["depth"]
