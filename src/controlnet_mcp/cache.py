"""Content-addressed disk cache for rendered analyses.

An analysis is identified by the digest of the source image bytes, the analysis
name, the processor's render version, the detect resolution, and an optional
variant such as a prompt digest, so repeating a request costs a file read
instead of a model run. Files live at
``output_dir / <digest> / <kind>-v<version>-<resolution>[-<variant>].png``.
"""

import hashlib
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

DIGEST_LENGTH = 16


def image_digest(data: bytes) -> str:
    """Return the cache key for image bytes.

    Args:
        data: Raw bytes of the source image.

    Returns:
        The first 16 hexadecimal characters of the sha256 digest, which is short
        enough for a readable directory name and wide enough to separate the
        images of one reference directory.
    """
    return hashlib.sha256(data).hexdigest()[:DIGEST_LENGTH]


class AnalysisCache:
    """Stores rendered analysis PNGs on disk, keyed by source image digest.

    Writes go to a temporary sibling file and are moved into place with
    ``os.replace``, so a concurrent reader either sees the previous file or the
    complete new one, never a partial write.
    """

    def __init__(self, output_dir: Path) -> None:
        """Bind the cache to its root directory.

        Args:
            output_dir: Directory that holds one subdirectory per source image.
                It is created on the first store, not here.
        """
        self.output_dir = output_dir

    def path_for(
        self,
        digest: str,
        kind: str,
        version: str,
        resolution: int,
        variant: str | None = None,
    ) -> Path:
        """Return the file path a rendered analysis occupies, whether or not it exists."""
        suffix = f"-{variant}" if variant else ""
        return self.output_dir / digest / f"{kind}-v{version}-{resolution}{suffix}.png"

    def get(
        self,
        digest: str,
        kind: str,
        version: str,
        resolution: int,
        variant: str | None = None,
    ) -> bytes | None:
        """Read a cached analysis.

        Returns:
            The stored PNG bytes, or None when the analysis has not been rendered
            at this version, resolution, and variant.
        """
        path = self.path_for(digest, kind, version, resolution, variant)
        if not path.is_file():
            logger.debug(
                "Cache miss for image %s, analysis %s at resolution %d", digest, kind, resolution
            )
            return None
        data = path.read_bytes()
        logger.debug(
            "Cache hit for image %s, analysis %s at resolution %d (%d bytes)",
            digest,
            kind,
            resolution,
            len(data),
        )
        return data

    def put(
        self,
        digest: str,
        kind: str,
        version: str,
        resolution: int,
        png_bytes: bytes,
        variant: str | None = None,
    ) -> Path:
        """Store a rendered analysis and return the path it now occupies.

        Args:
            digest: Digest of the source image bytes.
            kind: Analysis name, for example ``depth``.
            version: The processor's render version.
            resolution: Detect resolution the analysis was rendered at.
            png_bytes: Encoded PNG to store.
            variant: Extra key component, for example a prompt digest.
        """
        path = self.path_for(digest, kind, version, resolution, variant)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(png_bytes)
        try:
            os.replace(temporary, path)
        except OSError:
            temporary.unlink(missing_ok=True)
            raise
        logger.info(
            "Stored image %s, analysis %s at resolution %d (%d bytes)",
            digest,
            kind,
            resolution,
            len(png_bytes),
        )
        return path
