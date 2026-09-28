"""The perspective structure of an image's straight edges.

Straight segments are grouped into families of edges that run the same way in
the scene: the edges of a family meet at one vanishing point in the image, or
stay parallel when that point is too far away to tell from infinity. From the
families come the pieces of edges that share one line, and, when the evidence
allows, the horizon and an estimate of the camera. A value the evidence does
not support is withheld with the reason.

Pixel positions are in the frame the segments were detected in, origin top
left with y downward. Angles are in degrees with y upward, as they read on
the page: 0 runs to the image right and 90 runs up.
"""

import math
from dataclasses import dataclass

import numpy as np

from controlnet_mcp.evidence import Withheld

# Pixels, at detection resolution. The direction of a 20-pixel segment is
# uncertain by about atan(1/20), 3 degrees, from its endpoints each being placed
# to the nearest pixel.
MIN_DIRECTION_PIXELS = 20
# How many of the longest segments are intersected pairwise to propose vanishing
# points: 1770 proposals, enough that every family of at least
# ``MIN_FAMILY_SEGMENTS`` long edges proposes its own point many times over.
HYPOTHESIS_SEGMENTS = 60
# Degrees between a segment and the line from its midpoint to a vanishing point.
# Matches the direction uncertainty of the shortest segment that is counted.
INLIER_DEGREES = 3.0
# Six independent segments over-determine a two-parameter point three times.
MIN_FAMILY_SEGMENTS = 6
# Share of the total length of the counted segments. A family carrying less than
# a tenth of the straight-edge length is not one of the main directions.
MIN_FAMILY_LENGTH_SHARE = 0.1
# A scene has at most three mutually perpendicular directions.
MAX_FAMILIES = 3
# Image diagonals from the image center. At that distance edges across the frame
# converge by under atan(1/20), 3 degrees, which is below ``INLIER_DEGREES``, so
# converging and parallel cannot be told apart.
PARALLEL_DIAGONALS = 20
# Pixels. Each endpoint is placed to about 1 pixel, on each of the two segments
# being compared.
COLLINEAR_PIXELS = 2.0
# Degrees from the image vertical within which a family can stand for the
# verticals of the scene. A camera held by hand or on a tripod is rarely rolled
# or pitched enough to lean the verticals further at the image center.
VERTICAL_DEGREES = 10.0
# Share of the focal length. Pairs of families that disagree by more than a
# tenth do not describe one camera looking at perpendicular directions.
CAMERA_AGREEMENT = 0.1
# Pixels across the image between the two vanishing points the horizon is drawn
# through. Each point is placed to about a pixel, so closer than that the slope
# of the line through them comes from the placement error alone.
MIN_HORIZON_SPAN_PIXELS = 1.0
# Length of the intersection of two lines, below which they are one line. Lines
# have unit normals and offsets in pixels, so the intersection of two different
# lines is at least as long as the sine of the angle between them or, when they
# are parallel, the pixels between them. Only pieces of one line fall to the
# rounding error of the arithmetic.
SAME_LINE_LENGTH = 1e-9


@dataclass(frozen=True)
class LineFamily:
    """Segments that run the same way in the scene.

    Attributes:
        segment_indices: Indices into the segments given to the analysis.
        vanishing_point: Where the family's edges meet, in detection pixels,
            which may lie outside the frame. None when the family is parallel.
        direction_degrees: For a parallel family, the direction its edges run,
            from 0 up to 180. Otherwise the direction of the line from the
            image center toward the vanishing point, above -180 up to 180.
        scatter_degrees: The median angle between a segment and the line from
            its midpoint to the vanishing point, or to the parallel direction.
        length_share: The family's share of the total length of the counted
            segments.
    """

    segment_indices: tuple[int, ...]
    vanishing_point: tuple[float, float] | None
    direction_degrees: float
    scatter_degrees: float
    length_share: float


@dataclass(frozen=True)
class SharedLine:
    """Pieces of one family that lie on a single line.

    Attributes:
        family: Index of the family among the result's families.
        segment_indices: Indices into the segments given to the analysis.
        start: One outer end of the pieces, in detection pixels.
        end: The other outer end.
        gaps: The stretches between pieces, each as a pair of positions from 0
            at ``start`` to 1 at ``end``.
    """

    family: int
    segment_indices: tuple[int, ...]
    start: tuple[float, float]
    end: tuple[float, float]
    gaps: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Horizon:
    """Where the horizon crosses the frame.

    Attributes:
        left_y: Height of the horizon at the left border of the frame, as a
            fraction of the frame height from the top. It may lie outside 0
            to 1.
        right_y: The same at the right border.
        vertical_family: Index of the family taken for the verticals of the
            scene.
        source_families: Indices of the families whose vanishing points the
            horizon was drawn through.
        assumption: What had to be assumed to place it.
    """

    left_y: float
    right_y: float
    vertical_family: int
    source_families: tuple[int, ...]
    assumption: str


@dataclass(frozen=True)
class CameraEstimate:
    """The camera that the vanishing points are consistent with.

    Attributes:
        field_of_view_degrees: Angle of view across the width of the frame.
        pitch_degrees: How far the camera looks above level, negative below.
            None when no family stands for the verticals of the scene.
        roll_degrees: How far the verticals lean to the right at the top.
            None when no family stands for the verticals of the scene.
        pairs: How many pairs of converging families gave a focal length. One
            pair gives an estimate that nothing checks.
        disagreement: The spread of the focal lengths the pairs gave, as a
            share of their mean. Zero for one pair.
        source_families: Indices of the converging families the focal length
            came from.
        vertical_family: Index of the family the pitch and roll came from, or
            None when there is none.
        farthest_point_diagonals: Distance from the image center of the
            farthest vanishing point used, in image diagonals. The farther a
            point lies, the less precisely it is placed.
        assumption: What had to be assumed to estimate it.
    """

    field_of_view_degrees: float
    pitch_degrees: float | None
    roll_degrees: float | None
    pairs: int
    disagreement: float
    source_families: tuple[int, ...]
    vertical_family: int | None
    farthest_point_diagonals: float
    assumption: str


@dataclass(frozen=True)
class PerspectiveResult:
    """Everything the analysis found in one set of segments.

    Attributes:
        families: The families, the one carrying the most length first.
        unassigned: How many segments long enough to have a direction belong
            to no family.
        too_short: How many segments are shorter than
            ``MIN_DIRECTION_PIXELS`` and were left out of the grouping.
        shared_lines: Pieces lying on one line, within each family.
        horizon: The horizon, or why it could not be placed.
        camera: The camera estimate, or why it could not be made.
    """

    families: tuple[LineFamily, ...]
    unassigned: int
    too_short: int
    shared_lines: tuple[SharedLine, ...]
    horizon: Horizon | Withheld
    camera: CameraEstimate | Withheld


def analyze_perspective(
    segments: np.ndarray, width: int, height: int, cropped: bool
) -> PerspectiveResult:
    """Group straight segments into families and derive what the families support.

    Args:
        segments: Endpoint quadruples ``x0, y0, x1, y1`` in detection pixels,
            shaped (N, 4).
        width: Width in pixels of the frame the segments were detected in.
        height: Height in pixels of that frame.
        cropped: Whether the frame is a crop of the image, in which case its
            center is not the optical center and no camera is estimated.
    """
    segments = np.asarray(segments, dtype=np.float64).reshape(-1, 4)
    families = _find_families(segments, width, height)
    assigned = sum(len(family.segment_indices) for family in families)
    shared = [
        line
        for index, family in enumerate(families)
        for line in _shared_lines(segments, family, index)
    ]
    too_short = int((_lengths(segments) < MIN_DIRECTION_PIXELS).sum())
    return PerspectiveResult(
        families=tuple(families),
        unassigned=len(segments) - assigned - too_short,
        too_short=too_short,
        shared_lines=tuple(shared),
        horizon=_horizon(families, width, height),
        camera=_camera(families, width, height, cropped),
    )


def _lengths(segments: np.ndarray) -> np.ndarray:
    return np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])


def _find_families(segments: np.ndarray, width: int, height: int) -> list[LineFamily]:
    """Find the families one at a time, each among the segments the earlier ones left.

    The search is deterministic: the same segments always give the same
    families.
    """
    lengths = _lengths(segments)
    remaining = np.flatnonzero(lengths >= MIN_DIRECTION_PIXELS)
    total_length = float(lengths[remaining].sum())
    center = np.array([width / 2.0, height / 2.0])
    centered = segments - np.tile(center, 2)
    families: list[LineFamily] = []
    while len(families) < MAX_FAMILIES and len(remaining) >= MIN_FAMILY_SEGMENTS:
        members = _best_supported(centered[remaining], lengths[remaining])
        if members is None:
            break
        indices = remaining[members]
        share = float(lengths[indices].sum()) / total_length
        if share < MIN_FAMILY_LENGTH_SHARE:
            break
        point = _fit_point(centered[indices], math.hypot(width, height) / 2.0)
        families.append(_family(indices, point, centered, share, center, width, height))
        remaining = remaining[~members]
    return families


def _line_through(segments: np.ndarray) -> np.ndarray:
    """The line through each segment as ``a, b, c`` with ``a*x + b*y + c = 0`` and unit normal."""
    x0, y0, x1, y1 = segments.T
    a = y1 - y0
    b = x0 - x1
    norm = np.hypot(a, b)
    a, b = a / norm, b / norm
    return np.stack([a, b, -(a * x0 + b * y0)], axis=1)


def _angles_to(points: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Degrees between each segment and the line from its midpoint to each point.

    Points are homogeneous ``x, y, w``, so a point at infinity is a direction.

    Returns:
        Angles from 0 to 90, shaped (points, segments).
    """
    middle = (segments[:, :2] + segments[:, 2:]) / 2.0
    along = segments[:, 2:] - segments[:, :2]
    toward_x = points[:, 0, np.newaxis] - points[:, 2, np.newaxis] * middle[:, 0]
    toward_y = points[:, 1, np.newaxis] - points[:, 2, np.newaxis] * middle[:, 1]
    cross = np.abs(along[:, 0] * toward_y - along[:, 1] * toward_x)
    dot = np.abs(along[:, 0] * toward_x + along[:, 1] * toward_y)
    return np.degrees(np.arctan2(cross, dot))


def _best_supported(segments: np.ndarray, lengths: np.ndarray) -> np.ndarray | None:
    """The segments supporting the best proposed vanishing point, as a mask.

    Proposals are the intersections of every pair among the longest segments.
    The best is the one whose supporting segments carry the most length, among
    those supported by at least ``MIN_FAMILY_SEGMENTS``. None when no proposal
    has that many.
    """
    longest = np.argsort(-lengths, kind="stable")[:HYPOTHESIS_SEGMENTS]
    lines = _line_through(segments[longest])
    first, second = np.triu_indices(len(lines), k=1)
    proposals = np.cross(lines[first], lines[second])
    norms = np.linalg.norm(proposals, axis=1)
    # Two pieces of one line meet everywhere along it and propose nothing.
    proposals = proposals[norms > SAME_LINE_LENGTH]
    if len(proposals) == 0:
        return None
    supporting = _angles_to(proposals, segments) < INLIER_DEGREES
    enough = supporting.sum(axis=1) >= MIN_FAMILY_SEGMENTS
    if not enough.any():
        return None
    carried = np.where(enough, supporting @ lengths, -1.0)
    return supporting[int(np.argmax(carried))]


def _fit_point(segments: np.ndarray, scale: float) -> np.ndarray:
    """The point closest to the lines of all the segments, by least squares.

    The fit is made in coordinates divided by ``scale`` so the three
    components of a line are of like size, and in homogeneous form so that
    parallel lines fit a point at infinity.

    Returns:
        The point as homogeneous ``x, y, w`` in the pixel coordinates of the
        segments.
    """
    lines = _line_through(segments / scale)
    _, vectors = np.linalg.eigh(lines.T @ lines)
    x, y, w = vectors[:, 0]
    return np.array([x * scale, y * scale, w])


def _family(
    indices: np.ndarray,
    point: np.ndarray,
    centered: np.ndarray,
    share: float,
    center: np.ndarray,
    width: int,
    height: int,
) -> LineFamily:
    """Describe one family from its segments and the point fitted to them."""
    scatter = float(np.median(_angles_to(point[np.newaxis, :], centered[indices])[0]))
    reach = PARALLEL_DIAGONALS * math.hypot(width, height)
    distance = math.hypot(point[0], point[1])
    if distance > reach * abs(point[2]):
        vanishing_point = None
        direction = math.degrees(math.atan2(-point[1], point[0])) % 180.0
    else:
        x, y = point[0] / point[2], point[1] / point[2]
        vanishing_point = (float(x + center[0]), float(y + center[1]))
        direction = math.degrees(math.atan2(-y, x))
    return LineFamily(
        segment_indices=tuple(int(index) for index in indices),
        vanishing_point=vanishing_point,
        direction_degrees=direction,
        scatter_degrees=scatter,
        length_share=share,
    )


def _shared_lines(segments: np.ndarray, family: LineFamily, index: int) -> list[SharedLine]:
    """The groups of two or more of a family's segments that lie on one line.

    Two segments are joined when both ends of each lie within
    ``COLLINEAR_PIXELS`` of the other's line, and groups are closed under
    joining.
    """
    indices = np.array(family.segment_indices)
    members = segments[indices]
    lines = _line_through(members)
    ends = members.reshape(-1, 2, 2)
    distance = np.abs(
        lines[:, np.newaxis, np.newaxis, 0] * ends[np.newaxis, :, :, 0]
        + lines[:, np.newaxis, np.newaxis, 1] * ends[np.newaxis, :, :, 1]
        + lines[:, np.newaxis, np.newaxis, 2]
    )
    near = (distance <= COLLINEAR_PIXELS).all(axis=2)
    joined = near & near.T
    shared = []
    for group in _connected_groups(joined):
        if len(group) >= 2:
            shared.append(_shared_line(members[group], indices[group], index))
    return shared


def _connected_groups(joined: np.ndarray) -> list[list[int]]:
    """The groups of items connected through a symmetric matrix of joins."""
    unvisited = set(range(len(joined)))
    groups = []
    while unvisited:
        start = min(unvisited)
        unvisited.remove(start)
        group = [start]
        frontier = [start]
        while frontier:
            current = frontier.pop()
            for other in np.flatnonzero(joined[current]):
                if int(other) in unvisited:
                    unvisited.remove(int(other))
                    group.append(int(other))
                    frontier.append(int(other))
        groups.append(sorted(group))
    return groups


def _shared_line(pieces: np.ndarray, indices: np.ndarray, family: int) -> SharedLine:
    """Describe the line shared by pieces, from its outer ends and the gaps between pieces."""
    lengths = _lengths(pieces)
    longest = pieces[int(np.argmax(lengths))]
    along = (longest[2:] - longest[:2]) / lengths.max()
    ends = pieces.reshape(-1, 2)
    positions = ends @ along
    start = ends[int(np.argmin(positions))]
    end = ends[int(np.argmax(positions))]
    low, high = float(positions.min()), float(positions.max())
    spans = sorted(
        (float(min(pair)), float(max(pair))) for pair in positions.reshape(-1, 2).tolist()
    )
    gaps = []
    covered = spans[0][1]
    for span_start, span_end in spans[1:]:
        if span_start > covered:
            gaps.append(((covered - low) / (high - low), (span_start - low) / (high - low)))
        covered = max(covered, span_end)
    return SharedLine(
        family=family,
        segment_indices=tuple(int(index) for index in indices),
        start=(float(start[0]), float(start[1])),
        end=(float(end[0]), float(end[1])),
        gaps=tuple(gaps),
    )


def _lean_from_vertical(family: LineFamily) -> float:
    """Degrees the family's direction leans to the right of the image vertical at the top."""
    return 90.0 - family.direction_degrees % 180.0


def _vertical_family(families: list[LineFamily]) -> LineFamily | None:
    """The family closest to the image vertical, when one is within ``VERTICAL_DEGREES``."""
    upright = [
        family for family in families if abs(_lean_from_vertical(family)) <= VERTICAL_DEGREES
    ]
    if not upright:
        return None
    return min(upright, key=lambda family: abs(_lean_from_vertical(family)))


def _horizon(families: list[LineFamily], width: int, height: int) -> Horizon | Withheld:
    """Place the horizon from the families that run along the ground.

    A family near the image vertical is taken for the verticals of the scene,
    which makes the others horizontal in the scene, and the vanishing points
    of horizontal directions lie on the horizon.
    """
    vertical = _vertical_family(families)
    if vertical is None:
        return Withheld(
            f"no group runs within {VERTICAL_DEGREES:.0f}° of the image vertical, so none can "
            "be taken for the verticals of the scene"
        )
    upright = families.index(vertical)
    converging = [
        (index, family.vanishing_point)
        for index, family in enumerate(families)
        if family is not vertical and family.vanishing_point is not None
    ]
    sources = tuple(index for index, _ in converging)
    points = [point for _, point in converging]
    if len(points) >= 2:
        (x0, y0), (x1, y1) = points[0], points[1]
        if abs(x1 - x0) < MIN_HORIZON_SPAN_PIXELS:
            return Withheld("the two converging groups meet above one another, not side by side")
        slope = (y1 - y0) / (x1 - x0)
        return Horizon(
            left_y=(y0 - slope * x0) / height,
            right_y=(y0 + slope * (width - x0)) / height,
            vertical_family=upright,
            source_families=sources[:2],
            assumption="the near-vertical group is vertical in the scene",
        )
    if len(points) == 1 and vertical.vanishing_point is None:
        level = points[0][1] / height
        return Horizon(
            left_y=level,
            right_y=level,
            vertical_family=upright,
            source_families=sources,
            assumption=(
                "the near-vertical group is vertical in the scene and the image is not rolled"
            ),
        )
    if len(points) == 1:
        return Withheld(
            "only one group besides the verticals converges, and the verticals converge too, "
            "so the tilt of the horizon is not fixed"
        )
    return Withheld("no group besides the verticals converges to a point")


def _camera(
    families: list[LineFamily], width: int, height: int, cropped: bool
) -> CameraEstimate | Withheld:
    """Estimate the camera from vanishing points taken to be of perpendicular directions.

    With the optical center at the image center ``p``, two vanishing points
    ``v1`` and ``v2`` of perpendicular directions give the focal length in
    pixels by ``f² = -(v1 - p)·(v2 - p)``. Position, distance, and scale are
    not recoverable from vanishing points and are never reported.
    """
    if cropped:
        return Withheld("the image is a crop, and the optical center of a crop is unknown")
    center = np.array([width / 2.0, height / 2.0])
    sources = tuple(
        index for index, family in enumerate(families) if family.vanishing_point is not None
    )
    offsets = [np.array(families[index].vanishing_point) - center for index in sources]
    if len(offsets) < 2:
        return Withheld("fewer than two groups converge to a point")
    squares = [
        -float(offsets[first] @ offsets[second])
        for first in range(len(offsets))
        for second in range(first + 1, len(offsets))
    ]
    if min(squares) <= 0.0:
        return Withheld("the converging groups are not perpendicular in the scene")
    focal_lengths = np.sqrt(squares)
    focal = float(focal_lengths.mean())
    disagreement = float(focal_lengths.max() - focal_lengths.min()) / focal
    if disagreement > CAMERA_AGREEMENT:
        return Withheld(
            "pairs of groups disagree on the focal length by more than "
            f"{CAMERA_AGREEMENT:.0%}, so they are not three perpendicular directions seen by "
            "one camera"
        )
    for family in families:
        if family.vanishing_point is None and not _perpendicular(family, offsets, focal):
            return Withheld(
                "the parallel group is not perpendicular to the converging groups when the "
                "optical center is taken at the image center, as happens when an image was "
                "cropped, shifted, or had its verticals straightened"
            )
    vertical = _vertical_family(families)
    pitch, roll = _pitch_and_roll(vertical, center, focal)
    farthest = max(float(np.linalg.norm(offset)) for offset in offsets)
    return CameraEstimate(
        field_of_view_degrees=math.degrees(2.0 * math.atan(width / (2.0 * focal))),
        pitch_degrees=pitch,
        roll_degrees=roll,
        pairs=len(squares),
        disagreement=disagreement,
        source_families=sources,
        vertical_family=families.index(vertical) if vertical is not None else None,
        farthest_point_diagonals=farthest / math.hypot(width, height),
        assumption=(
            "the groups are perpendicular in the scene, the optical center is the image "
            "center, and the lens projects straight lines as straight"
        ),
    )


def _perpendicular(parallel: LineFamily, offsets: list[np.ndarray], focal: float) -> bool:
    """Whether a parallel family is perpendicular in the scene to every vanishing point.

    A parallel family runs across the view, along its image direction. A
    vanishing point at offset ``v`` from the optical center is the direction
    ``(v, f)`` in the scene.
    """
    angle = math.radians(parallel.direction_degrees)
    along = np.array([math.cos(angle), -math.sin(angle)])
    for offset in offsets:
        cosine = abs(float(offset @ along)) / math.hypot(float(np.linalg.norm(offset)), focal)
        if math.degrees(math.asin(min(cosine, 1.0))) > INLIER_DEGREES:
            return False
    return True


def _pitch_and_roll(
    vertical: LineFamily | None, center: np.ndarray, focal: float
) -> tuple[float | None, float | None]:
    """Pitch and roll of the camera from the family taken for the scene's verticals.

    Verticals that stay parallel mean a level camera. Verticals that meet
    above the image center mean the camera looks up, and below, down.
    """
    if vertical is None:
        return None, None
    roll = _lean_from_vertical(vertical)
    if vertical.vanishing_point is None:
        return 0.0, roll
    offset = np.array(vertical.vanishing_point) - center
    pitch = math.degrees(math.atan2(focal, float(np.linalg.norm(offset))))
    return (pitch if offset[1] < 0 else -pitch), roll
