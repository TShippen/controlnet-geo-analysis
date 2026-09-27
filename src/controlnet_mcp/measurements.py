"""Numbers read back off an analysis output, in a brief and a full form.

Each function measures one rendered analysis with numpy and cv2 alone, so nothing
here depends on torch or on the detector that produced the output. Coordinates are
fractions of the full reference image with the origin at the top left and two
decimals; an output rendered from a crop maps its positions through the crop.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from controlnet_mcp.evidence import Withheld
from controlnet_mcp.perspective import (
    INLIER_DEGREES,
    CameraEstimate,
    Horizon,
    LineFamily,
    PerspectiveResult,
)
from controlnet_mcp.regions import FULL_IMAGE, CropRegion

NEAR_THRESHOLD = 170
FAR_THRESHOLD = 85
BEYOND_RANGE_MAX = 2
EDGE_THRESHOLD = 128
FLAT_GRADIENT = 0.05
ORIENTATION_BIN = 0.25
ORIENTATION_MIN_SHARE = 0.05
# Degrees. The normals of a flat face may differ by about 3 degrees between
# neighbouring pixels (``FLAT_GRADIENT``), so a smaller turn or tilt is within the
# variation of a face that faces the camera squarely.
FACING_TOLERANCE_DEGREES = 5
# Neighbours that join two pixels into one face: the four sharing a side. Pixels
# touching only at a corner do not join, so two faces meeting at a point stay apart.
FACE_CONNECTIVITY = 4
LINE_LIMIT_BRIEF = 3
LINE_LIMIT_FULL = 12
# The color each group of the perspective analysis is drawn in, in the order the
# groups are found.
GROUP_COLORS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("red", (255, 0, 0)),
    ("green", (0, 255, 0)),
    ("blue", (0, 0, 255)),
)
GROUP_COLOR_NAMES = tuple(name for name, _ in GROUP_COLORS)
# Degrees. A group whose median edge sits at half the angle allowed to any edge
# fits its point loosely: lens distortion, curved edges, or mixed directions.
LOOSE_FIT_DEGREES = INLIER_DEGREES / 2


@dataclass(frozen=True)
class Measurement:
    """What one analysis output measured, at two lengths.

    Attributes:
        brief: A single sentence of about 120 characters at most. The
            perspective analysis is the exception: its brief form names every
            group, the horizon, and the camera estimate with their
            assumptions, and runs longer.
        full: The brief sentence extended with that analysis's extra detail.
    """

    brief: str
    full: str


EMPTY_MEASUREMENT = Measurement(brief="", full="")


def measure_depth(gray: np.ndarray, region: CropRegion = FULL_IMAGE) -> Measurement:
    """Split an 8-bit depth map, in which brighter is closer, into near, mid, far, and black.

    The rendering stretches each map between two depth percentiles and clips
    everything beyond the far one to black, so every map has a black share
    whatever the scene. That share is reported apart from far, which counts
    only pixels the map still orders. A pixel at or below ``BEYOND_RANGE_MAX``
    counts as black, allowing for the resize softening the clipped edge. The
    full form bounds the near region.

    Args:
        gray: The depth map.
        region: The part of the reference image the map was rendered from.
    """
    values = gray.astype(np.int16)
    near = values >= NEAR_THRESHOLD
    black = values <= BEYOND_RANGE_MAX
    near_share = _share(near)
    black_share = _share(black)
    far_share = _share((values < FAR_THRESHOLD) & ~black)
    mid_share = 100.0 - near_share - far_share - black_share
    brief = (
        f"Depth: near {near_share:.0f}%, mid {mid_share:.0f}%, far {far_share:.0f}% of pixels; "
        f"{black_share:.0f}% solid black, beyond the depth range."
    )
    if not near.any():
        return Measurement(brief=brief, full=f"{brief} No near region.")
    return Measurement(brief=brief, full=f"{brief} Near region {_bounding_box(near, region)}.")


def measure_normals(rgb: np.ndarray, region: CropRegion = FULL_IMAGE) -> Measurement:
    """Split a surface normal map into flat and curved area and describe its flat faces.

    A pixel counts as flat when its normal barely changes towards its right and
    lower neighbours. Flat normals are grouped by quantized bin, and each
    connected region of a group is one face, reported by how much of the image
    it covers and by the direction it faces. That direction comes from the mean
    normal of the face's pixels and is relative to the camera: how far the face
    is turned left or right of facing the camera and how far it is tilted up or
    down. The same face gets a different direction from another viewpoint. A
    face is a region of the map, not a recognized surface: sky and open
    background have even normals too and are reported like any other face.
    The brief form gives the largest face; the full form lists the faces
    covering at least ``ORIENTATION_MIN_SHARE``, largest first, each with its
    bounding box.

    Args:
        rgb: The normal map.
        region: The part of the reference image the map was rendered from.
    """
    normals = rgb.astype(np.float32) / 255.0 * 2.0 - 1.0
    flat = _neighbour_change(normals) < FLAT_GRADIENT
    total = flat.size
    flat_share = 100.0 * float(flat.sum()) / total
    faces = _flat_faces(normals, flat)
    largest = "largest flat face 0%"
    if faces:
        largest = f"largest flat face {faces[0].share(total):.0f}%, {faces[0].facing}"
    brief = f"Normals: flat {flat_share:.0f}%, curved {100.0 - flat_share:.0f}%; {largest}."
    threshold = f"{ORIENTATION_MIN_SHARE:.0%}"
    listed = [face for face in faces if face.count >= ORIENTATION_MIN_SHARE * total]
    if not listed:
        return Measurement(
            brief=brief, full=f"{brief} No flat face covers at least {threshold} of the image."
        )
    entries = [
        f"{face.share(total):.0f}% {face.facing}, {_bounding_box(face.mask, region)}"
        for face in listed[:LINE_LIMIT_FULL]
    ]
    return Measurement(
        brief=brief,
        full=(
            f"{brief} Flat faces covering at least {threshold} of the image, relative to the "
            "camera: " + "; ".join(entries) + "."
        ),
    )


def measure_edges(gray: np.ndarray, edges_are_dark: bool) -> Measurement:
    """Report how much of an 8-bit edge map is edge pixels.

    Args:
        gray: The edge map.
        edges_are_dark: Whether edges are drawn dark on a light ground rather
            than light on a dark ground.
    """
    values = gray.astype(np.int16)
    edges = values < EDGE_THRESHOLD if edges_are_dark else values > EDGE_THRESHOLD
    share = 100.0 * float(edges.sum()) / values.size
    return _both_forms(f"Edges cover {share:.1f}% of pixels.")


def measure_lines(
    segments: Sequence[Sequence[float]],
    width: int,
    height: int,
    region: CropRegion = FULL_IMAGE,
    long_only: bool = False,
) -> Measurement:
    """Count detected straight segments and name the longest of them.

    Args:
        segments: Endpoint quadruples ``x0, y0, x1, y1`` in pixels.
        width: Width in pixels of the image the segments were detected in.
        height: Height in pixels of that image.
        region: The part of the reference image that image was rendered from.
        long_only: Whether short segments were filtered out, which the
            wording then says.
    """
    kind = "long straight edge" if long_only else "straight edge"
    if len(segments) == 0:
        return _both_forms(f"No {kind}s found.")
    ordered = sorted(segments, key=_segment_length, reverse=True)
    noun = kind if len(ordered) == 1 else f"{kind}s"
    endpoints = [
        _endpoints(segment, width, height, region) for segment in ordered[:LINE_LIMIT_FULL]
    ]
    opening = f"{len(ordered)} {noun}; longest "
    return Measurement(
        brief=opening + ", ".join(endpoints[:LINE_LIMIT_BRIEF]) + ".",
        full=opening + ", ".join(endpoints) + ".",
    )


def measure_mask(mask: np.ndarray) -> Measurement:
    """Summarize a region mask as an area share and a bounding box, plus its centroid in full."""
    if not mask.any():
        return _both_forms("No region found.")
    height, width = mask.shape
    area = 100.0 * float(mask.sum()) / mask.size
    core = f"Region covers {area:.1f}% of the image; {_bounding_box(mask)}"
    rows, cols = np.nonzero(mask)
    centroid_x = (float(cols.mean()) + 0.5) / width
    centroid_y = (float(rows.mean()) + 0.5) / height
    return Measurement(
        brief=f"{core}.",
        full=f"{core}; centroid ({centroid_x:.2f}, {centroid_y:.2f}).",
    )


def measure_perspective(
    result: PerspectiveResult, width: int, height: int, region: CropRegion = FULL_IMAGE
) -> Measurement:
    """Report the groups of straight edges and what they support.

    Each group is named with the color it is drawn in, where its edges meet or
    the direction they run parallel, and how many edges it has. The horizon
    and the camera estimate follow, each with what was assumed to derive it,
    or with the reason it was withheld. A vanishing point may lie outside 0
    to 1, since edges often meet outside the frame. The full form adds each
    group's scatter and share of edge length, the count of unassigned edges,
    and the edges that share one line.

    Args:
        result: The perspective analysis.
        width: Width in pixels of the frame the analysis was made in.
        height: Height in pixels of that frame.
        region: The part of the reference image that frame was rendered from.
    """
    if not result.families:
        return _both_forms(
            "No group of straight edges converges or runs parallel; "
            f"{result.unassigned} straight edges unassigned."
        )
    brief_groups = []
    full_groups = []
    for index, family in enumerate(result.families):
        group = _group_phrase(index, family, width, height, region)
        brief_groups.append(group)
        full_groups.append(
            f"{group}, scatter {family.scatter_degrees:.1f}°, "
            f"{100.0 * family.length_share:.0f}% of edge length"
        )
    closing = (
        f"{_horizon_phrase(result.horizon, region)} {_camera_phrase(result.camera)}"
    )
    full = (
        "Perspective: " + "; ".join(full_groups) + f". {result.unassigned} edges unassigned. "
        f"{closing}{_shared_lines_phrase(result, width, height, region)}"
    )
    return Measurement(brief="Perspective: " + "; ".join(brief_groups) + f". {closing}", full=full)


def _group_phrase(
    index: int, family: LineFamily, width: int, height: int, region: CropRegion
) -> str:
    """One group's color, where it vanishes or how it runs, and its edge count."""
    if family.vanishing_point is None:
        where = f"parallel at {round(family.direction_degrees)}°"
    else:
        x, y = region.to_full(family.vanishing_point[0] / width, family.vanishing_point[1] / height)
        where = f"vanishes at ({x:.2f}, {y:.2f})"
    loose = ", loose fit" if family.scatter_degrees >= LOOSE_FIT_DEGREES else ""
    count = len(family.segment_indices)
    return f"group {index + 1} ({GROUP_COLOR_NAMES[index]}) {where}, {count} edges{loose}"


def _horizon_phrase(horizon: Horizon | Withheld, region: CropRegion) -> str:
    """Where the horizon crosses the borders of the full image, or why it is withheld."""
    if isinstance(horizon, Withheld):
        return f"Horizon withheld: {horizon.reason}."
    left_x, left_y = region.to_full(0.0, horizon.left_y)
    right_x, right_y = region.to_full(1.0, horizon.right_y)
    slope = (right_y - left_y) / (right_x - left_x)
    at_left = left_y - slope * left_x
    at_right = left_y + slope * (1.0 - left_x)
    return (
        f"Horizon crosses the left border at y {at_left:.2f} and the right border at y "
        f"{at_right:.2f}, assuming {horizon.assumption}."
    )


def _camera_phrase(camera: CameraEstimate | Withheld) -> str:
    """The camera estimate with its assumption, or why it is withheld."""
    if isinstance(camera, Withheld):
        return f"Camera estimate withheld: {camera.reason}."
    parts = [f"field of view {round(camera.field_of_view_degrees)}° across the width"]
    if camera.pitch_degrees is not None:
        parts.append(_signed_phrase(camera.pitch_degrees, "level", "looking", "up", "down"))
    if camera.roll_degrees is not None:
        roll = camera.roll_degrees
        parts.append(_signed_phrase(roll, "verticals upright", "verticals lean", "right", "left"))
    if camera.pairs == 1:
        support = "from 1 pair of converging groups, which nothing checks"
    else:
        support = (
            f"from {camera.pairs} pairs of converging groups that agree within "
            f"{math.ceil(100.0 * camera.disagreement)}%"
        )
    return f"Camera estimate {support}: " + ", ".join(parts) + f", assuming {camera.assumption}."


def _signed_phrase(degrees: float, zero: str, verb: str, positive: str, negative: str) -> str:
    """Phrase a signed angle by its direction, to whole degrees."""
    whole = round(degrees)
    if whole == 0:
        return zero
    return f"{verb} {abs(whole)}° {positive if whole > 0 else negative}"


def _shared_lines_phrase(
    result: PerspectiveResult, width: int, height: int, region: CropRegion
) -> str:
    """The longest lines shared by several edges, with a leading space, or nothing."""
    if not result.shared_lines:
        return ""
    ordered = sorted(
        result.shared_lines,
        key=lambda line: _segment_length((*line.start, *line.end)),
        reverse=True,
    )
    entries = []
    for line in ordered[:LINE_LIMIT_FULL]:
        ends = _endpoints((*line.start, *line.end), width, height, region)
        gaps = " and ".join(f"{start:.2f} to {end:.2f}" for start, end in line.gaps)
        between = f", gaps along it at {gaps}" if gaps else ""
        entries.append(
            f"group {line.family + 1} {ends} in {len(line.segment_indices)} pieces{between}"
        )
    return " Edges sharing one line: " + "; ".join(entries) + "."


def _share(mask: np.ndarray) -> float:
    """Percentage of the pixels that are true."""
    return 100.0 * float(mask.sum()) / mask.size


def _both_forms(text: str) -> Measurement:
    """A measurement whose brief and full forms are the same sentence."""
    return Measurement(brief=text, full=text)


def _bounding_box(mask: np.ndarray, region: CropRegion = FULL_IMAGE) -> str:
    """Phrase the normalized extent of the true pixels, with no trailing period."""
    height, width = mask.shape
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    x0, y0 = region.to_full(cols[0] / width, rows[0] / height)
    x1, y1 = region.to_full((cols[-1] + 1) / width, (rows[-1] + 1) / height)
    return (
        f"bounding box x {x0:.2f} to {x1:.2f}, y {y0:.2f} to {y1:.2f} (normalized, origin top-left)"
    )


def _neighbour_change(normals: np.ndarray) -> np.ndarray:
    """Magnitude of the change in the normal towards the right and lower neighbours.

    Pixels in the last column and the last row have no neighbour in that
    direction and contribute no change there.
    """
    right = np.zeros_like(normals)
    right[:, :-1] = normals[:, 1:] - normals[:, :-1]
    lower = np.zeros_like(normals)
    lower[:-1, :] = normals[1:, :] - normals[:-1, :]
    return np.sqrt((right * right).sum(axis=2) + (lower * lower).sum(axis=2))


@dataclass(frozen=True)
class _FlatFace:
    """One connected region of flat pixels sharing a quantized normal.

    Attributes:
        count: How many pixels the face has.
        facing: The direction of the face's mean normal, phrased relative to
            the camera.
        mask: The face's pixels in the map.
    """

    count: int
    facing: str
    mask: np.ndarray

    def share(self, total: int) -> float:
        """Percentage of a map of ``total`` pixels that the face covers."""
        return 100.0 * self.count / total


def _flat_faces(normals: np.ndarray, flat: np.ndarray) -> list[_FlatFace]:
    """The largest flat faces of a decoded normal map, largest first.

    Flat pixels are grouped by quantized normal, and each group is split into
    its connected regions, so two separate surfaces that happen to face the
    same way are two faces. Returns at most ``LINE_LIMIT_FULL`` faces, which is
    as many as any form of the measurement reports.
    """
    if not flat.any():
        return []
    _, inverse, counts = np.unique(
        np.round(normals[flat] / ORIENTATION_BIN), axis=0, return_inverse=True, return_counts=True
    )
    groups = np.full(flat.shape, -1, dtype=np.int64)
    groups[flat] = inverse.reshape(-1)
    regions: list[np.ndarray] = []
    for index in np.argsort(-counts, kind="stable"):
        # Groups come largest first, so once the kept regions are all at least as
        # large as a whole group, no later group can hold a larger region.
        if len(regions) == LINE_LIMIT_FULL and counts[index] <= regions[-1].sum():
            break
        regions.extend(_connected_regions(groups == index))
        regions.sort(key=lambda region: -int(region.sum()))
        del regions[LINE_LIMIT_FULL:]
    return [
        _FlatFace(count=int(mask.sum()), facing=_facing(normals[mask].mean(axis=0)), mask=mask)
        for mask in regions
    ]


def _connected_regions(mask: np.ndarray) -> list[np.ndarray]:
    """The largest connected regions of the true pixels, each as its own mask, largest first.

    Returns at most ``LINE_LIMIT_FULL`` regions.
    """
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=FACE_CONNECTIVITY
    )
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = np.argsort(-areas, kind="stable")[:LINE_LIMIT_FULL]
    return [labels == label + 1 for label in largest]


def _facing(normal: np.ndarray) -> str:
    """Phrase the direction of a decoded normal relative to the camera.

    The decoded red component is high for a face turned to the image left, so
    its negation points right. The turn is the angle left or right of facing
    the camera, and the tilt is the angle above or below level. An angle under
    ``FACING_TOLERANCE_DEGREES`` is left out.
    """
    right, up, toward = -float(normal[0]), float(normal[1]), float(normal[2])
    turn = round(math.degrees(math.atan2(right, toward)))
    tilt = round(math.degrees(math.atan2(up, math.hypot(right, toward))))
    parts = []
    if abs(turn) >= FACING_TOLERANCE_DEGREES:
        parts.append(f"turned {abs(turn)}° {'right' if turn > 0 else 'left'}")
    if abs(tilt) >= FACING_TOLERANCE_DEGREES:
        parts.append(f"tilted {abs(tilt)}° {'up' if tilt > 0 else 'down'}")
    if not parts:
        return "facing the camera"
    return ", ".join(parts)


def _segment_length(segment: Sequence[float]) -> float:
    x0, y0, x1, y1 = segment
    return math.hypot(x1 - x0, y1 - y0)


def _endpoints(segment: Sequence[float], width: int, height: int, region: CropRegion) -> str:
    """Both ends of one segment as coordinate pairs in fractions of the full image."""
    x0, y0, x1, y1 = segment
    start_x, start_y = region.to_full(_normalized(x0, width), _normalized(y0, height))
    end_x, end_y = region.to_full(_normalized(x1, width), _normalized(y1, height))
    return f"({start_x:.2f},{start_y:.2f})-({end_x:.2f},{end_y:.2f})"


def _normalized(value: float, extent: int) -> float:
    """One pixel coordinate as a fraction of an image extent, clipped to the frame.

    Line detection extrapolates endpoints past the border of the image, so a
    raw coordinate can fall outside the image.
    """
    return min(max(value / extent, 0.0), 1.0)
