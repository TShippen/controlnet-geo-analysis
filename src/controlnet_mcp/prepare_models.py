"""Download the v1 checkpoints into ``MODEL_DIR`` ahead of normal operation.

Run with ``uv run python -m controlnet_mcp.prepare_models``. Pass ``--check``
to report which checkpoints are missing without downloading anything.
"""

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path

from controlnet_mcp.checkpoints import (
    REQUIRED_CHECKPOINTS,
    CheckpointSpec,
    checkpoint_path,
    missing_checkpoints,
)
from controlnet_mcp.config import ConfigurationError, load_settings

logger = logging.getLogger(__name__)

Downloader = Callable[..., str]


def download_checkpoint(
    model_dir: Path, spec: CheckpointSpec, download: Downloader | None = None
) -> Path:
    """Ensure ``spec`` exists under ``model_dir``, downloading it when absent.

    Args:
        model_dir: Root of the checkpoint layout.
        spec: The checkpoint to fetch.
        download: A callable with ``hf_hub_download``'s keyword interface. Defaults to
            ``huggingface_hub.hf_hub_download``; tests inject a fake.

    Returns:
        The path of the checkpoint on disk.

    Raises:
        FileNotFoundError: When the download completed but the file is not where expected.
    """
    target = checkpoint_path(model_dir, spec)
    if target.is_file() and target.stat().st_size > 0:
        logger.info("Checkpoint present: %s", target)
        return target
    if download is None:
        from huggingface_hub import hf_hub_download

        download = hf_hub_download
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s from %s to %s", spec.filename, spec.repo_id, target.parent)
    download(repo_id=spec.repo_id, filename=spec.filename, local_dir=str(target.parent))
    if not target.is_file() or target.stat().st_size == 0:
        raise FileNotFoundError(f"Download of {spec.filename} did not produce {target}")
    logger.info("Downloaded %s (%d bytes)", target, target.stat().st_size)
    return target


def main(argv: list[str] | None = None) -> int:
    """Command entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Only report missing checkpoints.")
    parser.add_argument("--env-file", type=Path, default=None, help="Explicit .env file to load.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s")

    try:
        settings = load_settings(args.env_file)
    except ConfigurationError as exc:
        logger.error("%s", exc)
        return 2

    missing = missing_checkpoints(settings.model_dir, REQUIRED_CHECKPOINTS)
    if args.check:
        for spec in missing:
            logger.warning("Missing checkpoint: %s", checkpoint_path(settings.model_dir, spec))
        if missing:
            logger.error("%d checkpoint(s) missing under %s", len(missing), settings.model_dir)
            return 1
        logger.info(
            "All %d checkpoints present under %s", len(REQUIRED_CHECKPOINTS), settings.model_dir
        )
        return 0

    for spec in REQUIRED_CHECKPOINTS:
        download_checkpoint(settings.model_dir, spec)
    logger.info("All checkpoints ready under %s", settings.model_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
