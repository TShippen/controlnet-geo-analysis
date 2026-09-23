"""Tests for the checkpoint registry and the preparation command."""

from pathlib import Path

import pytest

from controlnet_mcp.checkpoints import (
    MOBILE_SAM_CHECKPOINT,
    NORMALBAE_CHECKPOINT,
    REQUIRED_CHECKPOINTS,
    ZOE_CHECKPOINT,
    checkpoint_path,
    missing_checkpoints,
)
from controlnet_mcp.prepare_models import download_checkpoint, main


def test_checkpoint_path_layout() -> None:
    model_dir = Path("/m")

    assert checkpoint_path(model_dir, ZOE_CHECKPOINT) == Path("/m/annotators/ZoeD_M12_N.pt")
    assert checkpoint_path(model_dir, MOBILE_SAM_CHECKPOINT) == Path("/m/mobile_sam/mobile_sam.pt")


def test_missing_checkpoints_lists_absent_files(tmp_path: Path) -> None:
    present = checkpoint_path(tmp_path, NORMALBAE_CHECKPOINT)
    present.parent.mkdir(parents=True)
    present.write_bytes(b"weights")

    missing = missing_checkpoints(tmp_path, REQUIRED_CHECKPOINTS)

    assert len(missing) == 5
    assert NORMALBAE_CHECKPOINT not in missing


def test_missing_checkpoints_treats_empty_file_as_missing(tmp_path: Path) -> None:
    empty = checkpoint_path(tmp_path, NORMALBAE_CHECKPOINT)
    empty.parent.mkdir(parents=True)
    empty.write_bytes(b"")

    assert NORMALBAE_CHECKPOINT in missing_checkpoints(tmp_path, [NORMALBAE_CHECKPOINT])


def test_download_skips_existing(tmp_path: Path) -> None:
    target = checkpoint_path(tmp_path, ZOE_CHECKPOINT)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"weights")
    calls: list[dict[str, str]] = []

    result = download_checkpoint(
        tmp_path, ZOE_CHECKPOINT, download=lambda **kw: calls.append(kw) or ""
    )

    assert result == target
    assert calls == []


def test_download_calls_hub_with_layout(tmp_path: Path) -> None:
    calls: list[dict[str, str]] = []

    def fake_download(**kwargs: str) -> str:
        calls.append(kwargs)
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.write_bytes(b"weights")
        return str(path)

    result = download_checkpoint(tmp_path, MOBILE_SAM_CHECKPOINT, download=fake_download)

    assert result == tmp_path / "mobile_sam" / "mobile_sam.pt"
    assert calls == [
        {
            "repo_id": "dhkim2810/MobileSAM",
            "filename": "mobile_sam.pt",
            "local_dir": str(tmp_path / "mobile_sam"),
        }
    ]


def test_download_raises_when_file_not_produced(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        download_checkpoint(tmp_path, ZOE_CHECKPOINT, download=lambda **kw: "")


def test_main_check_reports_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("REFERENCE_IMAGE_DIR", "MODEL_DIR", "OUTPUT_DIR"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "references").mkdir()
    (tmp_path / "models").mkdir()
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"REFERENCE_IMAGE_DIR={tmp_path / 'references'}\n"
        f"MODEL_DIR={tmp_path / 'models'}\n"
        f"OUTPUT_DIR={tmp_path / 'outputs'}\n"
    )

    assert main(["--check", "--env-file", str(env_file)]) == 1
