"""Tests for settings loading and directory validation."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from controlnet_mcp.config import ConfigurationError, apply_download_policy, load_settings

SETTING_VARIABLES = (
    "REFERENCE_IMAGE_DIR",
    "MODEL_DIR",
    "OUTPUT_DIR",
    "HF_HOME",
    "ALLOW_MODEL_DOWNLOADS",
    "DEFAULT_DETECT_RESOLUTION",
    "MAX_LOADED_MODELS",
    "DEVICE",
    "RESULT_MEASUREMENTS",
    "HF_HUB_OFFLINE",
)


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
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Hide the developer's settings from these tests and from every test that follows.

    ``load_settings`` exports the .env file it reads into the process
    environment, which other suites would otherwise inherit.
    """
    for name in SETTING_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    yield
    for name in SETTING_VARIABLES:
        os.environ.pop(name, None)


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

    with pytest.raises(ConfigurationError):
        load_settings(env_file)


def test_output_dir_is_created(tmp_path: Path) -> None:
    env_file = write_env(tmp_path)

    settings = load_settings(env_file)

    assert settings.output_dir.is_dir()


def test_resolution_out_of_range_rejected(tmp_path: Path) -> None:
    env_file = write_env(tmp_path, DEFAULT_DETECT_RESOLUTION="32")

    with pytest.raises(ConfigurationError):
        load_settings(env_file)


def test_result_measurements_default_is_brief(tmp_path: Path) -> None:
    settings = load_settings(write_env(tmp_path))

    assert settings.result_measurements == "brief"


def test_result_measurements_rejects_unknown_value(tmp_path: Path) -> None:
    env_file = write_env(tmp_path, RESULT_MEASUREMENTS="loud")

    with pytest.raises(ConfigurationError):
        load_settings(env_file)


def test_load_settings_exports_hf_home(tmp_path: Path) -> None:
    hf_home = tmp_path / "hf"
    env_file = write_env(tmp_path, HF_HOME=str(hf_home))

    settings = load_settings(env_file)

    assert os.environ["HF_HOME"] == str(hf_home)
    assert settings.hf_home == hf_home.resolve()


def test_relative_directories_resolve_against_the_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Launched from elsewhere, relative values still land beside the .env file."""
    env_file = write_env(
        tmp_path, REFERENCE_IMAGE_DIR="references", MODEL_DIR="models", OUTPUT_DIR="outputs"
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    settings = load_settings(env_file)

    assert settings.reference_image_dir == (tmp_path / "references").resolve()
    assert settings.model_dir == (tmp_path / "models").resolve()
    assert settings.output_dir == (tmp_path / "outputs").resolve()


def test_relative_hf_home_is_exported_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hugging Face reads HF_HOME itself, so the environment must hold the anchored path."""
    env_file = write_env(tmp_path, HF_HOME="hf")
    monkeypatch.chdir(tmp_path / "references")

    settings = load_settings(env_file)

    assert os.environ["HF_HOME"] == str(tmp_path / "hf")
    assert settings.hf_home == (tmp_path / "hf").resolve()


def test_environment_value_overrides_the_env_file(tmp_path: Path) -> None:
    """A host that sets a variable, such as an MCP config env block, wins over the .env file."""
    elsewhere = tmp_path / "elsewhere-models"
    elsewhere.mkdir()
    env_file = write_env(tmp_path, MODEL_DIR="models")
    os.environ["MODEL_DIR"] = str(elsewhere)

    settings = load_settings(env_file)

    assert settings.model_dir == elsewhere.resolve()


def test_without_an_env_file_relative_directories_use_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "references").mkdir()
    (tmp_path / "models").mkdir()
    monkeypatch.chdir(tmp_path)
    os.environ["REFERENCE_IMAGE_DIR"] = "references"
    os.environ["MODEL_DIR"] = "models"
    os.environ["OUTPUT_DIR"] = "outputs"

    settings = load_settings(tmp_path / "missing" / ".env")

    assert settings.model_dir == (tmp_path / "models").resolve()


def test_download_policy_exports_offline_flag(tmp_path: Path) -> None:
    settings = load_settings(write_env(tmp_path))

    apply_download_policy(settings)

    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_download_policy_leaves_environment_when_allowed(tmp_path: Path) -> None:
    settings = load_settings(write_env(tmp_path, ALLOW_MODEL_DOWNLOADS="true"))

    apply_download_policy(settings)

    assert "HF_HUB_OFFLINE" not in os.environ
