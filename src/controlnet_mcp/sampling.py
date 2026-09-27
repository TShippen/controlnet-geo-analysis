"""Values read off a decoded depth or normal map at chosen positions.

A position is a fraction of the full reference image with the origin at the
top left. Each sample reads a small window of the map around its position and
reports the value there together with how much the window varies, so a caller
can tell a reading on one surface from a reading that straddles two. Nothing
here knows how a map is encoded: the analysis that rendered the map decodes it
into a ``ValueMap`` first.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field

from controlnet_mcp.regions import CropRegion

# Pixels on each side of the sampled pixel. The 5-pixel window this gives is the
# smallest that straddles a one-pixel boundary from either side.
WINDOW_RADIUS = 2
# Depth levels. An eighth of the 0 to 255 level range within one 5-pixel window
# is more than a single receding surface produces.
BOUNDARY_LEVELS = 32
# Degrees. A face counted as flat changes by under 0.05 in its unit normal per
# pixel, about 3 degrees, so about 12 degrees across the 4 steps of a window; 20
# is well above that.
BOUNDARY_DEGREES = 20
# Depth levels between consecutive line samples. Large enough that the noise on
# one face does not trip it.
CHANGE_LEVELS = 24
# Degrees between consecutive line samples. Large enough that the noise on one
# face does not trip it.
CHANGE_DEGREES = 20
DEFAULT_LINE_SAMPLES = 32
MIN_LINE_SAMPLES = 2
MAX_LINE_SAMPLES = 512
MAX_POINTS = 64
POSITION_DECIMALS = 4
LEVEL_DECIMALS = 1
COMPONENT_DECIMALS = 3
DEGREE_DECIMALS = 1


class SamplingError(ValueError):
    """Raised when a sampling request is malformed or asks for a position that cannot be read."""


@dataclass(frozen=True)
class ValueMap:
    """A rendered analysis decoded into the values it encodes.

    Attributes:
        values: One value per pixel, shaped (height, width, channels), float32.
        beyond_range: Pixels that carry no value, shaped (height, width).
        vector: Whether each value is a unit direction rather than a level,
            which decides how spread and change are measured.
    """

    values: np.ndarray
    beyond_range: np.ndarray
    vector: bool


class Sample(BaseModel):
    """One reading of a value map, with the evidence for how far to trust it."""

    x: float = Field(description="Horizontal position as a fraction of the full image width.")
    y: float = Field(
        description="Vertical position as a fraction of the full image height, origin top-left."
    )
    value: list[float] | None = Field(
        description=(
            "The median value in a 5-pixel window around the position: one level for depth, or "
            "the components [right, up, toward the camera] of a unit direction for normals. "
            "Null when the position is beyond the depth range."
        )
    )
    spread: float = Field(
        description=(
            "How much the window varies: for depth the highest level minus the lowest, for "
            "normals the largest angle in degrees between any pixel and the reported direction."
        )
    )
    on_boundary: bool = Field(
        description=(
            "True when the spread is too large for one surface, so the sample sits between "
            "surfaces and a nearby position would read differently."
        )
    )
    beyond_range: bool = Field(
        description="True when more than half the window is beyond the depth range."
    )


class ValueChange(BaseModel):
    """A change in value between two consecutive samples on a line."""

    after_sample: int = Field(
        description=(
            "Index of the earlier sample: the change lies between samples after_sample and "
            "after_sample + 1."
        )
    )
    x: float = Field(description="Horizontal position of the midpoint between the two samples.")
    y: float = Field(description="Vertical position of the midpoint between the two samples.")
    size: float | None = Field(
        description=(
            "For depth the later level minus the earlier one, for normals the angle in degrees "
            "between the two directions. Null when one of the two samples is beyond the depth "
            "range."
        )
    )


class SampleReport(BaseModel):
    """The samples read from one analysis and, for a line, the changes between them.

    ``changes`` is empty when points were sampled, since separate points have
    no order to change along.
    """

    analysis: str = Field(description="The analysis the values were read from.")
    samples: list[Sample] = Field(description="One reading per position, in the order asked.")
    changes: list[ValueChange] = Field(
        description=(
            "For a line, each place the value changes between consecutive samples. A change "
            "brackets a boundary to within the sample spacing and does not locate it more finely."
        )
    )


def parse_points(values: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    """Validate a tool argument holding positions and return them as pairs.

    Raises:
        SamplingError: When there are no points or more than ``MAX_POINTS``,
            a point does not have two values, or a value is outside 0 to 1.
    """
    if not 1 <= len(values) <= MAX_POINTS:
        raise SamplingError(f"points must hold 1 to {MAX_POINTS} positions; got {len(values)}.")
    points = []
    for point in values:
        if len(point) != 2:
            raise SamplingError("each point must have two values: [x, y].")
        x, y = _checked_fractions(point, "point")
        points.append((x, y))
    return points


def parse_line(values: Sequence[float]) -> tuple[float, float, float, float]:
    """Validate a tool argument holding a line and return its two ends.

    Raises:
        SamplingError: When the list does not have four values, a value is
            outside 0 to 1, or both ends are the same position.
    """
    if len(values) != 4:
        raise SamplingError("line must have four values: [x0, y0, x1, y1].")
    x0, y0, x1, y1 = _checked_fractions(values, "line")
    if (x0, y0) == (x1, y1):
        raise SamplingError("line must run between two different positions.")
    return x0, y0, x1, y1


def sample_points(
    value_map: ValueMap, points: Sequence[tuple[float, float]], region: CropRegion
) -> list[Sample]:
    """Read the value map at each position.

    Args:
        value_map: The decoded map.
        points: Positions as fractions of the full reference image.
        region: The part of the reference image the map was rendered from.

    Raises:
        SamplingError: When a position lies outside ``region``.
    """
    return [_sample(value_map, x, y, region) for x, y in points]


def sample_line(
    value_map: ValueMap,
    start: tuple[float, float],
    end: tuple[float, float],
    count: int,
    region: CropRegion,
) -> tuple[list[Sample], list[ValueChange]]:
    """Read the value map at evenly spaced positions from ``start`` to ``end``, both included.

    Args:
        value_map: The decoded map.
        start: First position, as fractions of the full reference image.
        end: Last position, in the same coordinates.
        count: How many samples to take.
        region: The part of the reference image the map was rendered from.

    Returns:
        The samples in order, and each change between consecutive samples.

    Raises:
        SamplingError: When ``count`` is outside ``MIN_LINE_SAMPLES`` to
            ``MAX_LINE_SAMPLES`` or a position lies outside ``region``.
    """
    if not MIN_LINE_SAMPLES <= count <= MAX_LINE_SAMPLES:
        raise SamplingError(
            f"count must be between {MIN_LINE_SAMPLES} and {MAX_LINE_SAMPLES}; got {count}."
        )
    xs = np.linspace(start[0], end[0], count)
    ys = np.linspace(start[1], end[1], count)
    samples = [
        _sample(value_map, float(x), float(y), region) for x, y in zip(xs, ys, strict=True)
    ]
    changes = []
    for index in range(count - 1):
        change = _change_between(samples[index], samples[index + 1], index, value_map.vector)
        if change is not None:
            changes.append(change)
    return samples, changes


def _checked_fractions(values: Sequence[float], name: str) -> list[float]:
    """The values as floats, each required to lie between 0 and 1."""
    fractions = []
    for value in values:
        if not 0.0 <= float(value) <= 1.0:
            raise SamplingError(f"{name} coordinates must be between 0 and 1; got {value}.")
        fractions.append(float(value))
    return fractions


def _sample(value_map: ValueMap, x: float, y: float, region: CropRegion) -> Sample:
    """Read one window of the map.

    The value is taken over the window's pixels that carry one, while the
    spread is taken over every pixel in the window, so a window that reaches
    past the edge of the depth range reads as a boundary.
    """
    if not region.contains(x, y):
        raise SamplingError(
            f"Position ({x:.2f}, {y:.2f}) is outside the analyzed crop, "
            f"x {region.x0:.2f} to {region.x1:.2f}, y {region.y0:.2f} to {region.y1:.2f}."
        )
    values, beyond = _window(value_map, x, y, region)
    beyond_range = float(beyond.mean()) > 0.5
    usable = values[~beyond] if (~beyond).any() else values
    center = np.median(usable, axis=0)
    if value_map.vector:
        center = _unit(center)
        spread = float(_degrees_between(values, center).max())
        boundary = BOUNDARY_DEGREES
        decimals = COMPONENT_DECIMALS
    else:
        spread = float(values.max() - values.min())
        boundary = BOUNDARY_LEVELS
        decimals = LEVEL_DECIMALS
    value = None if beyond_range else [round(float(part), decimals) for part in center]
    return Sample(
        x=round(x, POSITION_DECIMALS),
        y=round(y, POSITION_DECIMALS),
        value=value,
        spread=round(spread, DEGREE_DECIMALS),
        on_boundary=spread > boundary,
        beyond_range=beyond_range,
    )


def _window(
    value_map: ValueMap, x: float, y: float, region: CropRegion
) -> tuple[np.ndarray, np.ndarray]:
    """The values and beyond-range flags around a position, one row per pixel.

    The window is cut short at the border of the map.
    """
    height, width = value_map.beyond_range.shape
    local_x, local_y = region.to_local(x, y)
    column = min(max(math.floor(local_x * width), 0), width - 1)
    row = min(max(math.floor(local_y * height), 0), height - 1)
    rows = slice(max(row - WINDOW_RADIUS, 0), row + WINDOW_RADIUS + 1)
    columns = slice(max(column - WINDOW_RADIUS, 0), column + WINDOW_RADIUS + 1)
    values = value_map.values[rows, columns]
    return values.reshape(-1, values.shape[-1]), value_map.beyond_range[rows, columns].reshape(-1)


def _change_between(
    earlier: Sample, later: Sample, index: int, vector: bool
) -> ValueChange | None:
    """The change between two consecutive samples, or None when the value holds.

    Passing into or out of the beyond-range area is always a change, and has
    no size because one side has no value.
    """
    if earlier.value is None and later.value is None:
        return None
    size: float | None = None
    if earlier.value is not None and later.value is not None:
        if vector:
            size = float(_degrees_between(np.array([later.value]), np.array(earlier.value))[0])
            threshold = CHANGE_DEGREES
        else:
            size = later.value[0] - earlier.value[0]
            threshold = CHANGE_LEVELS
        if abs(size) <= threshold:
            return None
        size = round(size, DEGREE_DECIMALS)
    return ValueChange(
        after_sample=index,
        x=round((earlier.x + later.x) / 2, POSITION_DECIMALS),
        y=round((earlier.y + later.y) / 2, POSITION_DECIMALS),
        size=size,
    )


def _unit(vector: np.ndarray) -> np.ndarray:
    """The vector scaled to unit length, or unchanged when it has no length to scale."""
    length = float(np.linalg.norm(vector))
    if length == 0.0:
        return vector
    return vector / length


def _degrees_between(vectors: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Angle in degrees between each row of unit vectors and one unit direction."""
    cosines = np.clip(vectors @ direction, -1.0, 1.0)
    return np.degrees(np.arccos(cosines))
