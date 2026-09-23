"""Registry of the checkpoint files the v1 processors need on disk.

Both the preparation command and the runtime model manager consult this
module, so the on-disk layout under ``MODEL_DIR`` is defined in one place.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

PREPARE_COMMAND = "uv run python -m controlnet_mcp.prepare_models"

ANNOTATORS_REPO = "lllyasviel/Annotators"
MOBILE_SAM_REPO = "dhkim2810/MobileSAM"


@dataclass(frozen=True)
class CheckpointSpec:
    """One checkpoint file: where it is published and where it lives under ``MODEL_DIR``."""

    repo_id: str
    filename: str
    relative_path: Path


ZOE_CHECKPOINT = CheckpointSpec(ANNOTATORS_REPO, "ZoeD_M12_N.pt", Path("annotators/ZoeD_M12_N.pt"))
NORMALBAE_CHECKPOINT = CheckpointSpec(ANNOTATORS_REPO, "scannet.pt", Path("annotators/scannet.pt"))
LINEART_CHECKPOINT = CheckpointSpec(
    ANNOTATORS_REPO, "sk_model.pth", Path("annotators/sk_model.pth")
)
LINEART_COARSE_CHECKPOINT = CheckpointSpec(
    ANNOTATORS_REPO, "sk_model2.pth", Path("annotators/sk_model2.pth")
)
MLSD_CHECKPOINT = CheckpointSpec(
    ANNOTATORS_REPO, "mlsd_large_512_fp32.pth", Path("annotators/mlsd_large_512_fp32.pth")
)
MOBILE_SAM_CHECKPOINT = CheckpointSpec(
    MOBILE_SAM_REPO, "mobile_sam.pt", Path("mobile_sam/mobile_sam.pt")
)

REQUIRED_CHECKPOINTS: tuple[CheckpointSpec, ...] = (
    ZOE_CHECKPOINT,
    NORMALBAE_CHECKPOINT,
    LINEART_CHECKPOINT,
    LINEART_COARSE_CHECKPOINT,
    MLSD_CHECKPOINT,
    MOBILE_SAM_CHECKPOINT,
)


def checkpoint_path(model_dir: Path, spec: CheckpointSpec) -> Path:
    """Absolute location of ``spec`` under ``model_dir``."""
    return model_dir / spec.relative_path


def missing_checkpoints(model_dir: Path, specs: Iterable[CheckpointSpec]) -> list[CheckpointSpec]:
    """Return the specs whose file is absent or empty under ``model_dir``."""
    missing: list[CheckpointSpec] = []
    for spec in specs:
        path = checkpoint_path(model_dir, spec)
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(spec)
    return missing
