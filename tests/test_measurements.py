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
from controlnet_mcp.regions import CropRegion

FACING_CAMERA = (128, 128, 255)
TURNED_LEFT = (218, 128, 218)


def test_depth_shares() -> None:
    """Near 200, 255, 255; mid 100; far 50; black 0: three, one, one, and one of six."""
    gray = np.array([[0, 100, 200], [255, 255, 50]], dtype=np.uint8)

    measurement = measure_depth(gray)

    assert "near 50%" in measurement.brief
    assert "mid 17%" in measurement.brief
    assert "far 17%" in measurement.brief
    assert "17% solid black" in measurement.brief


def test_depth_counts_softened_black_as_black() -> None:
    """Values up to 2 are the clipped region after resizing; 3 is ordinary far."""
    gray = np.array([[1, 2, 3, 3]], dtype=np.uint8)

    measurement = measure_depth(gray)

    assert "far 50%" in measurement.brief
    assert "50% solid black" in measurement.brief


def test_depth_full_bounds_the_near_region() -> None:
    gray = np.array([[0, 100, 200], [255, 255, 50]], dtype=np.uint8)

    full = measure_depth(gray).full

    assert "x 0.00 to 1.00" in full
    assert "y 0.00 to 1.00" in full


def test_depth_full_bounds_near_region_in_full_image_coordinates() -> None:
    """A crop of the right half: the bright left half of the crop is x 0.50 to 0.75 overall."""
    gray = np.zeros((100, 100), dtype=np.uint8)
    gray[:, :50] = 255

    full = measure_depth(gray, CropRegion(0.5, 0.0, 1.0, 1.0)).full

    assert "x 0.50 to 0.75" in full
    assert "y 0.00 to 1.00" in full


def test_depth_all_far() -> None:
    measurement = measure_depth(np.zeros((4, 4), dtype=np.uint8))

    assert "near 0%" in measurement.brief
    assert "far 0%" in measurement.brief
    assert "100% solid black" in measurement.brief
    assert "No near region." in measurement.full


def test_normals_solid_is_all_flat() -> None:
    rgb = np.full((4, 4, 3), FACING_CAMERA, dtype=np.uint8)

    measurement = measure_normals(rgb)

    assert "flat 100%" in measurement.brief
    assert "curved 0%" in measurement.brief
    assert "largest flat face 100%" in measurement.brief
    assert measurement.full.count("bounding box") == 1


def test_normals_two_faces() -> None:
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[:, :2] = FACING_CAMERA
    rgb[:, 2:] = TURNED_LEFT

    measurement = measure_normals(rgb)

    # The seam column differs from its right neighbour, so 4 of 16 pixels are curved,
    # which leaves the left face one column of 4 pixels and the right face two columns.
    assert "flat 75%" in measurement.brief
    assert "curved 25%" in measurement.brief
    assert "largest flat face 50%" in measurement.brief
    assert "50% turned 45° left" in measurement.full
    assert "25% facing the camera" in measurement.full


def test_normals_brief_names_largest_face_direction() -> None:
    """Red 218 and blue 218 decode to equal parts left and toward the camera: 45 degrees."""
    rgb = np.full((8, 8, 3), TURNED_LEFT, dtype=np.uint8)

    assert "turned 45° left" in measure_normals(rgb).brief


def test_normals_facing_camera_is_named() -> None:
    rgb = np.full((8, 8, 3), FACING_CAMERA, dtype=np.uint8)

    assert "facing the camera" in measure_normals(rgb).brief


def test_normals_direction_combines_turn_and_tilt() -> None:
    """Decoded (-0.5, 0.17, 0.85): right 0.5 over toward 0.85 is 30 degrees, up 0.17 is 10."""
    rgb = np.full((8, 8, 3), (64, 150, 236), dtype=np.uint8)

    assert "turned 30° right, tilted 10° up" in measure_normals(rgb).brief


def test_normals_full_lists_faces_with_direction_and_box() -> None:
    """Two halves of a map 400 wide.

    The seam column 199 is curved, so the left face covers columns 0 to 198:
    49.75% of the map, and a box ending at 199/400, both of which round to the
    half. The right face covers columns 200 to 399.
    """
    rgb = np.zeros((8, 400, 3), dtype=np.uint8)
    rgb[:, :200] = FACING_CAMERA
    rgb[:, 200:] = TURNED_LEFT

    full = measure_normals(rgb).full

    assert "50% facing the camera, bounding box x 0.00 to 0.50, y 0.00 to 1.00" in full
    assert "50% turned 45° left, bounding box x 0.50 to 1.00, y 0.00 to 1.00" in full


def test_normals_separate_regions_facing_one_way_are_separate_faces() -> None:
    """Two patches facing the camera with a turned band between them, on a map 400 wide.

    Columns 99 and 299 are seams and count as curved. The left patch covers
    columns 0 to 98 and the right patch columns 300 to 399, about 25% each, and
    the band covers columns 100 to 298, about 50%.
    """
    rgb = np.full((8, 400, 3), FACING_CAMERA, dtype=np.uint8)
    rgb[:, 100:300] = TURNED_LEFT

    full = measure_normals(rgb).full

    assert "50% turned 45° left, bounding box x 0.25 to 0.75" in full
    assert "25% facing the camera, bounding box x 0.00 to 0.25" in full
    assert "25% facing the camera, bounding box x 0.75 to 1.00" in full


def test_normals_largest_face_is_one_region_not_one_direction() -> None:
    """Three patches facing the camera are one direction but three faces.

    On a map 100 wide the patches are columns 0 to 19, 40 to 59, and 80 to 99,
    and the turned bands are columns 20 to 39 and 60 to 79. The last column of
    every region but the rightmost is a seam and counts as curved, so the
    rightmost patch keeps 20 columns and every other region keeps 19. Counted
    by direction, the patches would total 58%.
    """
    rgb = np.full((8, 100, 3), FACING_CAMERA, dtype=np.uint8)
    rgb[:, 20:40] = TURNED_LEFT
    rgb[:, 60:80] = TURNED_LEFT

    brief = measure_normals(rgb).brief

    assert "largest flat face 20%, facing the camera" in brief


def test_normals_face_box_maps_through_crop() -> None:
    """A face filling a crop of the right half spans x 0.50 to 1.00 of the full image."""
    rgb = np.full((8, 8, 3), FACING_CAMERA, dtype=np.uint8)

    full = measure_normals(rgb, CropRegion(0.5, 0.0, 1.0, 1.0)).full

    assert "bounding box x 0.50 to 1.00" in full


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


def test_lines_maps_endpoints_into_the_crop() -> None:
    measurement = measure_lines(
        [[0.0, 0.0, 100.0, 0.0]], 100, 100, CropRegion(0.5, 0.5, 1.0, 1.0)
    )

    assert "(0.50,0.50)-(1.00,0.50)" in measurement.brief


def test_lines_long_only_says_long() -> None:
    segments = [[0.0, 0.0, 50.0, 0.0], [0.0, 10.0, 20.0, 10.0]]

    measurement = measure_lines(segments, 100, 100, long_only=True)

    assert measurement.brief.startswith("2 long straight edges;")


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
