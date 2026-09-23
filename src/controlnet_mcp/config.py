"""Server settings loaded from the process environment and an optional .env file.

The .env file is exported into the process environment before settings are
read so that libraries which consult environment variables directly, such as
Hugging Face's ``HF_HOME``, observe the same values.
"""

import logging
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

MIN_RESOLUTION = 64
MAX_RESOLUTION = 2048

DeviceSetting = Literal["auto", "cpu", "cuda", "mps"]


class ConfigurationError(Exception):
    """Raised when the environment does not describe a usable server configuration."""


class Settings(BaseSettings):
    """Runtime configuration for the geometry analysis server.

    Directory fields are resolved to absolute paths. The reference and model
    directories must already exist; the output directory is created on load.
    """

    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    reference_image_dir: Path = Field(
        description="Directory holding the reference images agents may list, view, and analyze."
    )
    model_dir: Path = Field(description="Directory holding persistent processor checkpoints.")
    output_dir: Path = Field(description="Directory where generated analysis images are cached.")
    hf_home: Path | None = Field(
        default=None,
        description="Hugging Face cache directory used during model preparation.",
    )
    allow_model_downloads: bool = Field(
        default=False,
        description=(
            "Whether the running server may contact the Hugging Face hub. When false, "
            "HF_HUB_OFFLINE is exported at startup so any hub access fails instead of downloading."
        ),
    )
    default_detect_resolution: int = Field(
        default=512,
        ge=MIN_RESOLUTION,
        le=MAX_RESOLUTION,
        description="Detection resolution used when a tool call does not supply one.",
    )
    max_loaded_models: int = Field(
        default=1,
        ge=1,
        description="Maximum number of learned processors resident in memory at once.",
    )
    device: DeviceSetting = Field(
        default="auto",
        description="Torch device selection: auto picks cuda, then mps, then cpu.",
    )

    @model_validator(mode="after")
    def _resolve_and_check_directories(self) -> "Settings":
        self.reference_image_dir = self.reference_image_dir.expanduser().resolve()
        self.model_dir = self.model_dir.expanduser().resolve()
        self.output_dir = self.output_dir.expanduser().resolve()
        if self.hf_home is not None:
            self.hf_home = self.hf_home.expanduser().resolve()
        if not self.reference_image_dir.is_dir():
            raise ValueError(f"REFERENCE_IMAGE_DIR is not a directory: {self.reference_image_dir}")
        if not self.model_dir.is_dir():
            raise ValueError(f"MODEL_DIR is not a directory: {self.model_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self


def load_settings(env_file: Path | None = None) -> Settings:
    """Export the .env file into the environment and build validated settings.

    Args:
        env_file: Explicit .env path. When omitted, python-dotenv searches from
            the current working directory upward.

    Raises:
        ConfigurationError: When required variables are missing or invalid.
    """
    if env_file is not None:
        load_dotenv(env_file)
    else:
        load_dotenv()
    try:
        # Required fields are read from the environment by pydantic-settings.
        settings = Settings()  # type: ignore[call-arg]
    except ValidationError as exc:
        raise ConfigurationError(_describe_validation_error(exc)) from exc
    logger.info(
        "Loaded settings: references=%s models=%s outputs=%s device=%s max_loaded_models=%d",
        settings.reference_image_dir,
        settings.model_dir,
        settings.output_dir,
        settings.device,
        settings.max_loaded_models,
    )
    return settings


def apply_download_policy(settings: Settings) -> None:
    """Export ``HF_HUB_OFFLINE=1`` when the settings forbid downloads.

    Called by the server before any model is built, not by the preparation
    command, which exists to download.
    """
    if settings.allow_model_downloads:
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    logger.info("Model downloads disabled; HF_HUB_OFFLINE=1 exported")


def _describe_validation_error(exc: ValidationError) -> str:
    """Render pydantic's error list as one line per problem using environment variable names."""
    lines: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]).upper()
        message = error["msg"]
        if message.startswith("Value error, "):
            message = message[len("Value error, ") :]
        lines.append(f"{location}: {message}" if location else message)
    return "Invalid configuration: " + "; ".join(lines)
