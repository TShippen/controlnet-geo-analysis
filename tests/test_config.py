"""Tests for settings loading and directory validation."""

import os
from pathlib import Path

import pytest

from controlnet_mcp.config import ConfigurationError, load_settings


def write_env(tmp_path: Path, **overrides: str) -> Path:
    references = tmp_path / "references"
    models = tmp_path / "models"
    references.mkdir(exist_ok=True)
    models.mkdir(exist_ok=True)
    values = {
        "REFERENCE_IMAGE_DIR": str(references),
        "MODEL_DIR": str(models),
        "OUTPUT_DIR": str(tmp_path / "outputs"),
    }
    values.update(overrides)
    env_file = tmp_path / ".env"
    env_file.write_text("".join(f"{key}={value}\n" for key, value in values.items()))
    return env_file


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "REFERENCE_IMAGE_DIR",
        "MODEL_DIR",
        "OUTPUT_DIR",
        "HF_HOME",
        "ALLOW_MODEL_DOWNLOADS",
        "DEFAULT_DETECT_RESOLUTION",
        "MAX_LOADED_MODELS",
        "DEVICE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_load_settings_reads_env_file(tmp_path: Path) -> None:
    env_file = write_env(tmp_path)

    settings = load_settings(env_file)

    assert settings.reference_image_dir == (tmp_path / "references").resolve()
    assert settings.model_dir == (tmp_path / "models").resolve()
    assert settings.output_dir == (tmp_path / "outputs").resolve()
    assert settings.default_detect_resolution == 512
    assert settings.max_loaded_models == 1
    assert settings.device == "auto"
    assert settings.allow_model_downloads is False


def test_missing_reference_dir_raises(tmp_path: Path) -> None:
    env_file = write_env(tmp_path, REFERENCE_IMAGE_DIR=str(tmp_path / "nowhere"))

    with pytest.raises(ConfigurationError, match="REFERENCE_IMAGE_DIR"):
        load_settings(env_file)


def test_output_dir_is_created(tmp_path: Path) -> None:
    env_file = write_env(tmp_path)

    settings = load_settings(env_file)

    assert settings.output_dir.is_dir()


def test_resolution_out_of_range_rejected(tmp_path: Path) -> None:
    env_file = write_env(tmp_path, DEFAULT_DETECT_RESOLUTION="32")

    with pytest.raises(ConfigurationError, match="DEFAULT_DETECT_RESOLUTION"):
        load_settings(env_file)


def test_load_settings_exports_hf_home(tmp_path: Path) -> None:
    hf_home = tmp_path / "hf"
    env_file = write_env(tmp_path, HF_HOME=str(hf_home))

    settings = load_settings(env_file)

    assert os.environ["HF_HOME"] == str(hf_home)
    assert settings.hf_home == hf_home.resolve()
