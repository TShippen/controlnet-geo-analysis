"""Tests for the measurements computed from analysis outputs.

Every expected value is hand-counted from the tiny array built in the test.
"""

import dataclasses

import numpy as np

from conftest import camera_scene_test_segments
from controlnet_mcp.evidence import Withheld
from controlnet_mcp.measurements import (
    measure_depth,
    measure_edges,
    measure_lines,
    measure_mask,
    measure_normals,
    measure_perspective,
)
from controlnet_mcp.perspective import (
    CameraEstimate,
    Horizon,
    LineFamily,
    PerspectiveResult,
    analyze_perspective,
)
from controlnet_mcp.regions import CropRegion

FACING_CAMERA = (128, 128, 255)
TURNED_LEFT = (218, 128, 218)
NOTHING_WITHHELD = Withheld("not part of this test")


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


def perspective_test_result(
    families: tuple[LineFamily, ...] = (),
    camera: CameraEstimate | Withheld = NOTHING_WITHHELD,
) -> PerspectiveResult:
    """A perspective result built by hand, with the horizon withheld."""
    return PerspectiveResult(
        families=families,
        unassigned=0,
        shared_lines=(),
        horizon=NOTHING_WITHHELD,
        camera=camera,
    )


def perspective_test_family(point: tuple[float, float]) -> LineFamily:
    """A family of eight segments meeting at ``point``."""
    return LineFamily(
        segment_indices=tuple(range(8)),
        vanishing_point=point,
        direction_degrees=0.0,
        scatter_degrees=0.0,
        length_share=1.0,
    )


def test_perspective_brief_names_families() -> None:
    """In a frame 512 wide, x 948.8 is 1.85 of the width and x 25.1 is 0.05."""
    result = analyze_perspective(camera_scene_test_segments(), 512, 512, cropped=False)

    brief = measure_perspective(result, 512, 512).brief

    assert "vanishes at (1.85, 0.50)" in brief
    assert "vanishes at (0.05, 0.50)" in brief
    assert "parallel at 90°" in brief


def test_perspective_brief_reports_the_camera_with_its_assumption() -> None:
    """The camera scene has focal length 400: a field of view of 65 degrees, level, upright."""
    result = analyze_perspective(camera_scene_test_segments(), 512, 512, cropped=False)

    brief = measure_perspective(result, 512, 512).brief

    assert "Camera estimate from 1 pair of converging groups, which nothing checks:" in brief
    assert "field of view 65° across the width, level, verticals upright, assuming" in brief
    assert "Horizon crosses the left border at y 0.50 and the right border at y 0.50" in brief


def test_perspective_vanishing_point_maps_through_crop() -> None:
    """Local x 1024 is twice the frame width; in a crop of the right half that is 1.50."""
    result = perspective_test_result((perspective_test_family((1024.0, 256.0)),))

    brief = measure_perspective(result, 512, 512, CropRegion(0.5, 0.0, 1.0, 1.0)).brief

    assert "vanishes at (1.50, 0.50)" in brief


def test_perspective_horizon_maps_through_crop() -> None:
    """A horizon falling from 0.25 to 0.75 across a crop of the right half.

    The crop spans x 0.5 to 1 of the full image, so the line falls 1.0 per
    unit of full width and crosses the full image's left border at -0.25.
    """
    result = PerspectiveResult(
        families=(perspective_test_family((1024.0, 256.0)),),
        unassigned=0,
        shared_lines=(),
        horizon=Horizon(left_y=0.25, right_y=0.75, assumption="a test"),
        camera=NOTHING_WITHHELD,
    )

    brief = measure_perspective(result, 512, 512, CropRegion(0.5, 0.0, 1.0, 1.0)).brief

    assert "left border at y -0.25 and the right border at y 0.75" in brief


def test_perspective_text_gives_withheld_reason() -> None:
    result = perspective_test_result(
        (perspective_test_family((1024.0, 256.0)),),
        camera=Withheld("families are not perpendicular"),
    )

    brief = measure_perspective(result, 512, 512).brief

    assert "Camera estimate withheld: families are not perpendicular." in brief


def test_perspective_camera_reports_how_well_its_pairs_agree() -> None:
    """A disagreement of 3.2% is reported rounded up, as agreeing within 4%."""
    camera = CameraEstimate(
        field_of_view_degrees=60.0,
        pitch_degrees=None,
        roll_degrees=None,
        pairs=3,
        disagreement=0.032,
        assumption="a test",
    )
    result = perspective_test_result((perspective_test_family((1024.0, 256.0)),), camera)

    brief = measure_perspective(result, 512, 512).brief

    assert (
        "Camera estimate from 3 pairs of converging groups that agree within 4%: "
        "field of view 60° across the width, assuming a test."
    ) in brief


def test_perspective_loose_fit_is_named() -> None:
    """A median of 2 degrees is past half of the 3 degrees allowed to any edge."""
    family = dataclasses.replace(perspective_test_family((1024.0, 256.0)), scatter_degrees=2.0)

    brief = measure_perspective(perspective_test_result((family,)), 512, 512).brief

    assert "8 edges, loose fit" in brief


def test_perspective_full_lists_shared_lines() -> None:
    """Three pieces on row 100 of a 512 frame, columns 0 to 260, plus five more rows."""
    rows = [(100.0, 0.0, 50.0), (100.0, 80.0, 130.0), (100.0, 200.0, 260.0)]
    rows += [(140.0, 100.0, 160.0), (180.0, 90.0, 170.0), (220.0, 120.0, 220.0)]
    rows += [(260.0, 60.0, 170.0), (300.0, 200.0, 320.0)]
    segments = np.array([[x0, y, x1, y] for y, x0, x1 in rows])
    result = analyze_perspective(segments, 512, 512, cropped=False)

    measurement = measure_perspective(result, 512, 512)

    assert "(0.00,0.20)-(0.51,0.20) in 3 pieces, gaps along it at 0.19 to 0.31" in measurement.full
    assert "pieces" not in measurement.brief


def test_perspective_without_families_says_so() -> None:
    measurement = measure_perspective(perspective_test_result(), 512, 512)

    assert measurement.brief == (
        "No group of straight edges converges or runs parallel; 0 straight edges unassigned."
    )


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
