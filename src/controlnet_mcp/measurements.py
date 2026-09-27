"""Numbers read back off an analysis output, in a brief and a full form.

Each function measures one rendered analysis with numpy alone, so nothing here
depends on torch or on the detector that produced the output. Coordinates are
fractions of the full reference image with the origin at the top left and two
decimals; an output rendered from a crop maps its positions through the crop.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

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
LINE_LIMIT_BRIEF = 3
LINE_LIMIT_FULL = 12


@dataclass(frozen=True)
class Measurement:
    """What one analysis output measured, at two lengths.

    Attributes:
        brief: A single sentence of about 120 characters at most.
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
    lower neighbours. Flat normals are grouped by quantized bin, and each group
    is one face, reported by how much of the image it covers and by the
    direction it faces. That direction comes from the mean normal of the
    group's pixels and is relative to the camera: how far the face is turned
    left or right of facing the camera and how far it is tilted up or down. The
    same face gets a different direction from another viewpoint. The brief
    form gives the largest face; the full form lists the faces covering at
    least ``ORIENTATION_MIN_SHARE``, largest first, each with its bounding box.

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
    """The flat pixels sharing one quantized normal.

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

    Returns at most ``LINE_LIMIT_FULL`` faces, which is as many as any form of
    the measurement reports.
    """
    if not flat.any():
        return []
    flat_normals = normals[flat]
    _, inverse, counts = np.unique(
        np.round(flat_normals / ORIENTATION_BIN), axis=0, return_inverse=True, return_counts=True
    )
    inverse = inverse.reshape(-1)
    faces = []
    for index in np.argsort(-counts, kind="stable")[:LINE_LIMIT_FULL]:
        members = inverse == index
        mask = np.zeros(flat.shape, dtype=bool)
        mask[flat] = members
        faces.append(
            _FlatFace(
                count=int(counts[index]),
                facing=_facing(flat_normals[members].mean(axis=0)),
                mask=mask,
            )
        )
    return faces


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
