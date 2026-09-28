"""Straight edges of two images, brought into one frame and paired.

The second image is mapped into the frame of the first, either by a flat
transform fitted from features the two images share or, when the caller says
the images already share a frame, by scaling alone. Edges that lie close in
direction and position are then paired, and each pair reports how far apart
its edges are. Neither image is treated as the correct one.

A fitted transform is one flat mapping. When the two views differ, what is
left after it mixes real differences with the parallax of depth, and nothing
here can tell them apart. A pair means two edges lie close together, not that
they are the same physical edge.

Pixel positions are in the frame each image was detected in, origin top left.
"""

import math
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from controlnet_mcp.evidence import Withheld

AlignMode = Literal["fit", "none"]

# Ratio of the best descriptor distance to the second best. The value customary
# for this kind of feature, which rejects a match nearly as good as its runner-up.
RATIO_TEST = 0.75
# Pixels a matched feature may land from where the transform puts it. Features
# are located to about a pixel in each image, and the fit adds about as much.
ALIGN_PIXELS = 3.0
# Twelve separate points over-determine the 8 parameters of the transform three
# times.
MIN_INLIERS = 12
ALIGNMENT_GRID = 4
# Share of the cells of a 4x4 grid over the first image that hold a feature the
# transform explains: 2 of 16. One textured patch cannot stand for the whole image.
MIN_COVERAGE = 0.125
# Share of the first fit's features. A second, different transform explaining
# half as many features is a rival reading of how the images relate.
AMBIGUITY_SHARE = 0.5
# Share by which two aspect ratios may differ and still be one frame, which
# allows for the detection sizes of both images being rounded.
ASPECT_TOLERANCE = 0.01
# Degrees between two edges of a pair. Detected directions are good to about
# 3 degrees on a short edge, and a little is left for what alignment leaves over.
PAIR_DEGREES = 5.0
# Share of the longer side of the frame that the edges of a pair may lie apart.
# Wide enough to measure a misplaced edge, narrow enough to keep to its neighbourhood.
PAIR_DISTANCE_SHARE = 0.05
# Pixels. Each endpoint is placed to about 1 pixel, on each of the two edges
# being compared.
COLLINEAR_PIXELS = 2.0
# Pixels from the origin past which an edge is left out of the drawing. A
# transform can send an endpoint toward infinity, and the drawing takes
# coordinates as 32-bit integers, which a million pixels stays well inside.
DRAWABLE_PIXELS = 1e6
BACKGROUND_BRIGHTNESS = 0.4
FIRST_COLOR = (0, 255, 255)
SECOND_COLOR = (255, 0, 255)
OFFSET_COLOR = (255, 255, 0)

_WITHHELD_REASON = (
    "too few matching features, or features in too small a part of the image: the views are "
    "too different, unrelated, or too plain to align"
)


class ComparisonError(ValueError):
    """Raised when two images cannot be compared in the way that was asked."""


@dataclass(frozen=True)
class Alignment:
    """How the second image maps into the frame of the first, and what supports it.

    Attributes:
        transform: Maps second-image detection pixels to first-image detection
            pixels, as a 3x3 matrix.
        fitted: Whether the transform was fitted from matching features. When
            False the caller asserted a shared frame and the counts are zero.
        matched: How many features of the two images matched.
        inliers: How many of those the transform explains.
        coverage: Share of the cells of a grid over the first image that hold
            an explained feature.
        second_fit_inliers: How many of the features left over a second,
            different transform explains.
        ambiguous: Whether that second transform explains nearly as many.
    """

    transform: np.ndarray
    fitted: bool
    matched: int
    inliers: int
    coverage: float
    second_fit_inliers: int
    ambiguous: bool


@dataclass(frozen=True)
class EdgePair:
    """Two edges, one from each image, that lie close in direction and position.

    Attributes:
        first_index: Index into the first image's segments.
        second_index: Index into the second image's segments.
        offset: The displacement of the second edge's midpoint from the first
            edge's line, perpendicular to that line, in first-image pixels.
        end_offsets: How much further the second edge runs than the first at
            the first edge's start and at its end, along the first edge, in
            first-image pixels. Negative where the second edge stops short.
    """

    first_index: int
    second_index: int
    offset: tuple[float, float]
    end_offsets: tuple[float, float]


@dataclass(frozen=True)
class Pairing:
    """Which edges of two images were paired, and what became of the rest.

    Every segment is in exactly one place: a pair, the unmatched edges of its
    image, or the pieces on a matched line of its image.

    Attributes:
        pairs: The paired edges.
        unmatched_first: First-image segments with no partner.
        unmatched_second: Second-image segments with no partner.
        on_matched_line_first: First-image segments with no partner that lie
            on the line of a paired first-image segment: a broken piece of a
            matched edge, not a missing edge.
        on_matched_line_second: The same among the second image's segments.
    """

    pairs: tuple[EdgePair, ...]
    unmatched_first: tuple[int, ...]
    unmatched_second: tuple[int, ...]
    on_matched_line_first: tuple[int, ...]
    on_matched_line_second: tuple[int, ...]


def align_images(first: np.ndarray, second: np.ndarray) -> Alignment | Withheld:
    """Fit the flat transform that maps the second image onto the first.

    Args:
        first: The first image as 8-bit grayscale.
        second: The second image as 8-bit grayscale.

    Returns:
        The alignment, or why the images could not be aligned.
    """
    matches = _matched_points(first, second)
    if matches is None or len(matches[0]) < MIN_INLIERS:
        return Withheld(_WITHHELD_REASON)
    first_points, second_points = matches
    fit = _fit_transform(second_points, first_points)
    if fit is None:
        return Withheld(_WITHHELD_REASON)
    transform, explained = fit
    inliers = int(explained.sum())
    coverage = _coverage(first_points[explained], first.shape[1], first.shape[0])
    if inliers < MIN_INLIERS or coverage < MIN_COVERAGE:
        return Withheld(_WITHHELD_REASON)
    second_fit = _second_fit_inliers(
        second_points[~explained], first_points[~explained], transform, second.shape
    )
    return Alignment(
        transform=transform,
        fitted=True,
        matched=len(first_points),
        inliers=inliers,
        coverage=coverage,
        second_fit_inliers=second_fit,
        ambiguous=second_fit >= AMBIGUITY_SHARE * inliers,
    )


def identity_alignment(first_size: tuple[int, int], second_size: tuple[int, int]) -> Alignment:
    """The alignment of two images the caller says share one frame.

    Args:
        first_size: Width and height of the first image's detection frame.
        second_size: Width and height of the second image's detection frame.

    Raises:
        ComparisonError: When the aspect ratios differ by more than
            ``ASPECT_TOLERANCE``, so the images cannot share one frame.
    """
    first_aspect = first_size[0] / first_size[1]
    second_aspect = second_size[0] / second_size[1]
    if abs(first_aspect - second_aspect) > ASPECT_TOLERANCE * first_aspect:
        raise ComparisonError(
            "The images have different proportions "
            f"({first_size[0]}x{first_size[1]} and {second_size[0]}x{second_size[1]} when "
            "analyzed), so they do not share one frame. Use align fit, or crop them to the "
            "same proportions."
        )
    scale = np.diag([first_size[0] / second_size[0], first_size[1] / second_size[1], 1.0])
    return Alignment(
        transform=scale,
        fitted=False,
        matched=0,
        inliers=0,
        coverage=0.0,
        second_fit_inliers=0,
        ambiguous=False,
    )


def map_segments(segments: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Map endpoint quadruples through a 3x3 transform."""
    segments = np.asarray(segments, dtype=np.float64).reshape(-1, 4)
    if len(segments) == 0:
        return segments
    points = segments.reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, transform.astype(np.float64)).reshape(-1, 4)


def pair_edges(
    first: np.ndarray, second: np.ndarray, transform: np.ndarray, frame_size: tuple[int, int]
) -> Pairing:
    """Pair the edges of two images that lie close together in the first image's frame.

    Two edges can pair when their directions differ by under ``PAIR_DEGREES``,
    each one's midpoint lies within ``PAIR_DISTANCE_SHARE`` of the longer side
    of the frame from the other's line, and they run alongside each other for
    some of their length. The pairs are those in which each edge is the
    other's closest, so giving the images in the other order finds the same
    pairs.

    Args:
        first: The first image's segments, shaped (N, 4), in its pixels.
        second: The second image's segments, shaped (M, 4), in its pixels.
        transform: Maps second-image pixels to first-image pixels.
        frame_size: Width and height of the first image's detection frame.
    """
    first = np.asarray(first, dtype=np.float64).reshape(-1, 4)
    mapped = map_segments(second, transform)
    distance = _pair_distances(first, mapped, PAIR_DISTANCE_SHARE * max(frame_size))
    pairs = []
    if distance.size:
        best_second = distance.argmin(axis=1)
        best_first = distance.argmin(axis=0)
        for index, partner in enumerate(best_second):
            if np.isfinite(distance[index, partner]) and best_first[partner] == index:
                pairs.append(_edge_pair(first, mapped, index, int(partner)))
    paired_first = [pair.first_index for pair in pairs]
    paired_second = [pair.second_index for pair in pairs]
    on_line_first = _on_matched_lines(first, paired_first)
    on_line_second = _on_matched_lines(mapped, paired_second)
    return Pairing(
        pairs=tuple(pairs),
        unmatched_first=_remaining(len(first), paired_first, on_line_first),
        unmatched_second=_remaining(len(mapped), paired_second, on_line_second),
        on_matched_line_first=on_line_first,
        on_matched_line_second=on_line_second,
    )


def render_comparison(
    first_image: np.ndarray,
    second_image: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    alignment: Alignment | Withheld,
    pairing: Pairing | None,
) -> np.ndarray:
    """Draw the edges of both images over the dimmed first image.

    First-image edges are cyan, second-image edges, mapped into the first
    frame, are magenta, and a yellow line joins the midpoints of each pair.
    When the alignment is withheld nothing can be mapped, so the two images
    are drawn side by side, each dimmed under its own edges, with no pairs.

    Args:
        first_image: The first image as RGB at its detection size.
        second_image: The second image as RGB at its detection size.
        first: The first image's segments.
        second: The second image's segments.
        alignment: How the second image maps onto the first, or why not.
        pairing: The pairs, or None when the alignment is withheld.
    """
    canvas = _dimmed(first_image)
    _draw(canvas, first, FIRST_COLOR)
    if isinstance(alignment, Withheld) or pairing is None:
        panel = _dimmed(second_image)
        _draw(panel, second, SECOND_COLOR)
        height = max(canvas.shape[0], panel.shape[0])
        side_by_side = np.zeros((height, canvas.shape[1] + panel.shape[1], 3), dtype=np.uint8)
        side_by_side[: canvas.shape[0], : canvas.shape[1]] = canvas
        side_by_side[: panel.shape[0], canvas.shape[1] :] = panel
        return side_by_side
    mapped = map_segments(second, alignment.transform)
    _draw(canvas, mapped, SECOND_COLOR)
    first = np.asarray(first, dtype=np.float64).reshape(-1, 4)
    for pair in pairing.pairs:
        start = (first[pair.first_index, :2] + first[pair.first_index, 2:]) / 2.0
        end = (mapped[pair.second_index, :2] + mapped[pair.second_index, 2:]) / 2.0
        _draw(canvas, np.array([[*start, *end]]), OFFSET_COLOR)
    return canvas


def _matched_points(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Positions of the features the two images share, in each image, or None with none.

    The detector reports one spot several times when it finds several
    orientations there. Matches between the same two spots are kept once, to
    the nearest pixel, so every match counted is a separate point.
    """
    detector = cv2.SIFT.create()
    first_keys, first_descriptors = detector.detectAndCompute(first, None)
    second_keys, second_descriptors = detector.detectAndCompute(second, None)
    if first_descriptors is None or second_descriptors is None:
        return None
    if len(first_keys) < 2 or len(second_keys) < 2:
        return None
    candidates = cv2.BFMatcher(cv2.NORM_L2).knnMatch(second_descriptors, first_descriptors, k=2)
    kept = [
        best
        for best, runner_up in (pair for pair in candidates if len(pair) == 2)
        if best.distance < RATIO_TEST * runner_up.distance
    ]
    if not kept:
        return None
    points = np.array(
        [[*first_keys[match.trainIdx].pt, *second_keys[match.queryIdx].pt] for match in kept],
        dtype=np.float64,
    )
    _, distinct = np.unique(np.round(points), axis=0, return_index=True)
    points = points[np.sort(distinct)]
    return points[:, :2], points[:, 2:]


def _fit_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """The transform taking ``source`` points to ``target`` points, and which points it explains.

    The random sampling inside the fit is seeded, so the same points always
    give the same transform.
    """
    if len(source) < 4:
        return None
    cv2.setRNGSeed(0)
    transform, explained = cv2.findHomography(source, target, cv2.USAC_MAGSAC, ALIGN_PIXELS)
    if transform is None or explained is None:
        return None
    return transform, explained.reshape(-1).astype(bool)


def _coverage(points: np.ndarray, width: int, height: int) -> float:
    """Share of the cells of a grid over the image that hold at least one point."""
    columns = np.clip((points[:, 0] * ALIGNMENT_GRID / width).astype(int), 0, ALIGNMENT_GRID - 1)
    rows = np.clip((points[:, 1] * ALIGNMENT_GRID / height).astype(int), 0, ALIGNMENT_GRID - 1)
    return len(set(zip(rows.tolist(), columns.tolist(), strict=True))) / ALIGNMENT_GRID**2


def _second_fit_inliers(
    source: np.ndarray, target: np.ndarray, first_transform: np.ndarray, shape: tuple[int, ...]
) -> int:
    """How many of the features the first fit left over a second, different fit explains.

    A second fit that moves the corners of the image to within ``ALIGN_PIXELS``
    of where the first put them is the same alignment found again, and counts
    for nothing.
    """
    fit = _fit_transform(source, target)
    if fit is None:
        return 0
    transform, explained = fit
    height, width = shape[:2]
    corners = np.array([[0.0, 0.0, width, 0.0], [width, height, 0.0, height]])
    moved = map_segments(corners, transform) - map_segments(corners, first_transform)
    if float(np.abs(moved).max()) <= ALIGN_PIXELS:
        return 0
    return int(explained.sum())


def _unit_directions(segments: np.ndarray) -> np.ndarray:
    along = segments[:, 2:] - segments[:, :2]
    return along / np.linalg.norm(along, axis=1, keepdims=True)


def _midpoints(segments: np.ndarray) -> np.ndarray:
    return (segments[:, :2] + segments[:, 2:]) / 2.0


def _line_distances(segments: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Distance of each point from the line of each segment, shaped (segments, points)."""
    along = _unit_directions(segments)
    normal = np.stack([-along[:, 1], along[:, 0]], axis=1)
    offset = points[np.newaxis, :, :] - segments[:, np.newaxis, :2]
    return np.abs((offset * normal[:, np.newaxis, :]).sum(axis=2))


def _pair_distances(first: np.ndarray, mapped: np.ndarray, limit: float) -> np.ndarray:
    """How far apart each first edge and each mapped edge lie, infinite where they cannot pair.

    The distance is the larger of the two distances from one edge's midpoint
    to the other's line, which is the same whichever image comes first.
    """
    if len(first) == 0 or len(mapped) == 0:
        return np.zeros((len(first), len(mapped)))
    first_along = _unit_directions(first)
    mapped_along = _unit_directions(mapped)
    cosine = np.abs(first_along @ mapped_along.T)
    turned = np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))
    apart = np.maximum(
        _line_distances(first, _midpoints(mapped)),
        _line_distances(mapped, _midpoints(first)).T,
    )
    can_pair = (turned < PAIR_DEGREES) & (apart < limit) & _run_alongside(first, mapped)
    return np.where(can_pair, apart, np.inf)


def _run_alongside(first: np.ndarray, mapped: np.ndarray) -> np.ndarray:
    """Whether each first edge and each mapped edge overlap along their shared direction.

    The shared direction is the mean of the two edges' directions, so the
    answer is the same whichever image comes first.
    """
    first_along = _unit_directions(first)[:, np.newaxis, :]
    mapped_along = _unit_directions(mapped)[np.newaxis, :, :]
    same_way = np.sign((first_along * mapped_along).sum(axis=2, keepdims=True))
    shared = first_along + np.where(same_way == 0, 1.0, same_way) * mapped_along
    shared = shared / np.linalg.norm(shared, axis=2, keepdims=True)
    first_ends = np.stack(
        [
            (first[:, np.newaxis, :2] * shared).sum(axis=2),
            (first[:, np.newaxis, 2:] * shared).sum(axis=2),
        ]
    )
    mapped_ends = np.stack(
        [
            (mapped[np.newaxis, :, :2] * shared).sum(axis=2),
            (mapped[np.newaxis, :, 2:] * shared).sum(axis=2),
        ]
    )
    overlap = np.minimum(first_ends.max(axis=0), mapped_ends.max(axis=0)) - np.maximum(
        first_ends.min(axis=0), mapped_ends.min(axis=0)
    )
    return overlap > 0.0


def _edge_pair(first: np.ndarray, mapped: np.ndarray, index: int, partner: int) -> EdgePair:
    """Measure one pair along and across the first edge."""
    start, end = first[index, :2], first[index, 2:]
    length = float(np.linalg.norm(end - start))
    along = (end - start) / length
    normal = np.array([-along[1], along[0]])
    middle = (mapped[partner, :2] + mapped[partner, 2:]) / 2.0
    across = float((middle - start) @ normal) * normal
    reach = [float((end_point - start) @ along) for end_point in mapped[partner].reshape(2, 2)]
    return EdgePair(
        first_index=index,
        second_index=partner,
        offset=(float(across[0]), float(across[1])),
        end_offsets=(-min(reach), max(reach) - length),
    )


def _on_matched_lines(segments: np.ndarray, paired: list[int]) -> tuple[int, ...]:
    """The unpaired segments lying on the line of a paired segment of the same image."""
    if not paired or len(segments) == len(paired):
        return ()
    lines = segments[paired]
    near_start = _line_distances(lines, segments[:, :2]) <= COLLINEAR_PIXELS
    near_end = _line_distances(lines, segments[:, 2:]) <= COLLINEAR_PIXELS
    on_a_line = (near_start & near_end).any(axis=0)
    return tuple(
        index for index in range(len(segments)) if on_a_line[index] and index not in set(paired)
    )


def _remaining(count: int, paired: list[int], on_line: tuple[int, ...]) -> tuple[int, ...]:
    """The indices that are neither paired nor on a matched line."""
    taken = set(paired) | set(on_line)
    return tuple(index for index in range(count) if index not in taken)


def _dimmed(image: np.ndarray) -> np.ndarray:
    return (image.astype(np.float32) * BACKGROUND_BRIGHTNESS).astype(np.uint8)


def _draw(canvas: np.ndarray, segments: np.ndarray, color: tuple[int, int, int]) -> None:
    """Draw segments a pixel wide onto an RGB canvas.

    A segment mapped to a position that is not finite, or too far off to be a
    pixel coordinate, is skipped.
    """
    for x0, y0, x1, y1 in np.asarray(segments, dtype=np.float64).reshape(-1, 4):
        ends = (x0, y0, x1, y1)
        if not all(math.isfinite(value) and abs(value) < DRAWABLE_PIXELS for value in ends):
            continue
        cv2.line(canvas, (round(x0), round(y0)), (round(x1), round(y1)), color, 1)
