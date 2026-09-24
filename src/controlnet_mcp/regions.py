"""A rectangular crop of a reference image, in normalized coordinates.

An analysis can run on part of an image instead of all of it. The crop is
given as fractions of the full image with the origin at the top left, and
coordinates measured inside the crop are mapped back to fractions of the full
image, so every reported position lines up with the original picture.
"""

import hashlib
import math
from dataclasses import dataclass

MIN_CROP_PIXELS = 16
CROP_DECIMALS = 4
CROP_DIGEST_LENGTH = 8


class CropError(ValueError):
    """Raised when a crop is malformed, too small, or given to an analysis that cannot take one."""


@dataclass(frozen=True)
class CropRegion:
    """A rectangle ``x0, y0, x1, y1`` as fractions of the full image, origin top-left."""

    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_list(cls, values: list[float]) -> "CropRegion":
        """Validate a tool argument and build a region.

        Coordinates are rounded to ``CROP_DECIMALS`` places, so crops that
        differ only by float noise share one cache entry.

        Raises:
            CropError: When the list does not have four values, a value is
                outside 0 to 1, or the corners are not top-left then bottom-right.
        """
        if len(values) != 4:
            raise CropError("crop must have four values: [x0, y0, x1, y1].")
        for value in values:
            if not 0.0 <= float(value) <= 1.0:
                raise CropError(f"crop coordinates must be between 0 and 1; got {value}.")
        x0, y0, x1, y1 = (round(float(value), CROP_DECIMALS) + 0.0 for value in values)
        if x0 >= x1 or y0 >= y1:
            raise CropError("crop must run from the top-left corner to the bottom-right.")
        return cls(x0, y0, x1, y1)

    def digest(self) -> str:
        """Short stable identifier for cache file names."""
        canonical = f"crop={self.x0!r},{self.y0!r},{self.x1!r},{self.y1!r}"
        return hashlib.sha256(canonical.encode()).hexdigest()[:CROP_DIGEST_LENGTH]

    def pixel_box(self, width: int, height: int) -> tuple[int, int, int, int]:
        """The pixel rectangle ``left, top, right, bottom`` covering the region, rounded outward.

        Raises:
            CropError: When the rectangle is narrower or shorter than
                ``MIN_CROP_PIXELS`` source pixels.
        """
        left = math.floor(self.x0 * width)
        top = math.floor(self.y0 * height)
        right = math.ceil(self.x1 * width)
        bottom = math.ceil(self.y1 * height)
        if right - left < MIN_CROP_PIXELS or bottom - top < MIN_CROP_PIXELS:
            raise CropError(
                f"crop covers {right - left}x{bottom - top} pixels of a {width}x{height} image; "
                f"it must be at least {MIN_CROP_PIXELS} pixels on each side."
            )
        return left, top, right, bottom

    def snapped(self, width: int, height: int) -> "CropRegion":
        """The region aligned to the pixels ``pixel_box`` actually covers."""
        left, top, right, bottom = self.pixel_box(width, height)
        return CropRegion(left / width, top / height, right / width, bottom / height)

    def to_full(self, x: float, y: float) -> tuple[float, float]:
        """Map a point given as fractions of this region to fractions of the full image."""
        return self.x0 + x * (self.x1 - self.x0), self.y0 + y * (self.y1 - self.y0)


FULL_IMAGE = CropRegion(0.0, 0.0, 1.0, 1.0)
