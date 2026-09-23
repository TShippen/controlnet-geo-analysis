"""Tests for the measurements computed from analysis outputs.

Every expected value is hand-counted from the tiny array built in the test.
"""

import numpy as np

from controlnet_mcp.measurements import (
    measure_depth,
    measure_edges,
    measure_lines,
    measure_mask,
    measure_normals,
)


def test_depth_shares() -> None:
    gray = np.array([[0, 100, 200], [255, 255, 50]], dtype=np.uint8)

    measurement = measure_depth(gray)

    assert "near 50%" in measurement.brief
    assert "mid 17%" in measurement.brief
    assert "far 33%" in measurement.brief


def test_depth_full_bounds_the_near_region() -> None:
    gray = np.array([[0, 100, 200], [255, 255, 50]], dtype=np.uint8)

    full = measure_depth(gray).full

    assert "x 0.00 to 1.00" in full
    assert "y 0.00 to 1.00" in full


def test_depth_all_far() -> None:
    measurement = measure_depth(np.zeros((4, 4), dtype=np.uint8))

    assert "near 0%" in measurement.brief
    assert "far 100%" in measurement.brief
    assert "No near region." in measurement.full


def test_normals_solid_is_all_flat() -> None:
    rgb = np.full((4, 4, 3), 128, dtype=np.uint8)

    measurement = measure_normals(rgb)

    assert "flat 100%" in measurement.brief
    assert "curved 0%" in measurement.brief
    assert "largest flat face 100%" in measurement.brief
    assert "1 flat orientation covers" in measurement.full


def test_normals_two_faces() -> None:
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[:, :2] = (200, 100, 100)
    rgb[:, 2:] = (60, 180, 90)

    measurement = measure_normals(rgb)

    # The seam column differs from its right neighbour, so 4 of 16 pixels are curved.
    assert "flat 75%" in measurement.brief
    assert "curved 25%" in measurement.brief
    assert "largest flat face 50%" in measurement.brief
    assert "2 flat orientations cover" in measurement.full


def test_normals_gradient_is_curved() -> None:
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.linspace(0, 255, 32, dtype=np.uint8)

    measurement = measure_normals(rgb)

    # Only the last column has no right neighbour to differ from: 31 of 32 columns curve.
    assert "curved 97%" in measurement.brief


def test_edges_density_white_on_black() -> None:
    gray = np.zeros((4, 4), dtype=np.uint8)
    gray[0, 0] = 255
    gray[1, 2] = 255

    measurement = measure_edges(gray, edges_are_dark=False)

    assert "12.5%" in measurement.brief
    assert measurement.full == measurement.brief


def test_edges_density_dark_on_white() -> None:
    gray = np.full((4, 4), 255, dtype=np.uint8)
    gray[3, :] = 0

    measurement = measure_edges(gray, edges_are_dark=True)

    assert "25.0%" in measurement.brief


def test_lines_reports_longest_first() -> None:
    segments = [(0.0, 10.0, 10.0, 10.0), (0.0, 0.0, 30.0, 0.0), (0.0, 20.0, 20.0, 20.0)]

    brief = measure_lines(segments, 100, 100).brief

    assert "3 straight edges" in brief
    assert brief.index("(0.00,0.00)-(0.30,0.00)") < brief.index("(0.00,0.20)-(0.20,0.20)")
    assert brief.index("(0.00,0.20)-(0.20,0.20)") < brief.index("(0.00,0.10)-(0.10,0.10)")


def test_lines_caps_full_list() -> None:
    segments = [(0.0, float(index), float(index + 1), float(index)) for index in range(15)]

    measurement = measure_lines(segments, 100, 100)

    assert "15 straight edges" in measurement.brief
    assert measurement.brief.count(")-(") == 3
    assert measurement.full.count(")-(") == 12


def test_lines_single_segment_uses_singular() -> None:
    brief = measure_lines([(0.0, 0.0, 30.0, 0.0)], 100, 100).brief

    assert brief == "1 straight edge; longest (0.00,0.00)-(0.30,0.00)."


def test_lines_clips_endpoints_to_the_frame() -> None:
    """The line detector extrapolates endpoints past the border of the image."""
    segments = [(-5.0, 54.0, 101.0, 54.0)]

    brief = measure_lines(segments, 100, 100).brief

    assert "(0.00,0.54)-(1.00,0.54)" in brief


def test_lines_empty() -> None:
    measurement = measure_lines([], 100, 100)

    assert measurement.brief == "No straight edges found."
    assert measurement.full == "No straight edges found."


def test_mask_reports_bbox_and_area() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 1:5] = True

    brief = measure_mask(mask).brief

    assert "25.0%" in brief
    assert "x 0.12 to 0.62" in brief
    assert "y 0.25 to 0.75" in brief


def test_mask_full_has_centroid() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 1:5] = True

    full = measure_mask(mask).full

    assert "centroid (0.38, 0.50)" in full


def test_mask_empty() -> None:
    measurement = measure_mask(np.zeros((4, 4), dtype=bool))

    assert measurement.brief == "No region found."
    assert measurement.full == "No region found."
