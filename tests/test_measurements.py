"""Tests for the measurements computed from analysis outputs.

Every expected value is hand-counted from the tiny array built in the test.
"""

import dataclasses

import numpy as np

from conftest import camera_scene_test_segments
from controlnet_mcp.comparison import Alignment, EdgePair, Pairing, identity_alignment
from controlnet_mcp.evidence import Withheld
from controlnet_mcp.measurements import (
    comparison_outcome,
    measure_comparison,
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
        too_short=0,
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

    assert "field of view 65° across the width, level, verticals upright, assuming" in brief


def test_perspective_names_the_groups_behind_each_derived_value() -> None:
    """In the camera scene the verticals are found first, then the right and left sets.

    The verticals carry the most length, 800 pixels against 704, so they are
    group 1, and the two converging sets are groups 2 and 3.
    """
    result = analyze_perspective(camera_scene_test_segments(), 512, 512, cropped=False)

    brief = measure_perspective(result, 512, 512).brief

    assert (
        "Horizon, drawn through the vanishing points of group 2 (green) and group 3 (blue), "
        "crosses the left border at y 0.50 and the right border at y 0.50"
    ) in brief
    assert "the near-vertical group is group 1 (red)." in brief
    assert (
        "Camera estimate from 1 pair of converging groups, group 2 (green) and group 3 (blue), "
        "which nothing checks:"
    ) in brief
    assert "The tilt is read from group 1 (red)." in brief


def test_perspective_gives_the_distance_of_the_farthest_point_used() -> None:
    """The right point lies 692.8 pixels from the center of a frame whose diagonal is 724.1."""
    result = analyze_perspective(camera_scene_test_segments(), 512, 512, cropped=False)

    brief = measure_perspective(result, 512, 512).brief

    assert "The farthest vanishing point used lies 1.0 image diagonals from the image" in brief
    assert "the farther a point lies the less precisely it is placed" in brief


def test_perspective_says_how_a_parallel_direction_is_measured() -> None:
    result = analyze_perspective(camera_scene_test_segments(), 512, 512, cropped=False)

    brief = measure_perspective(result, 512, 512).brief

    assert "with 0° running to the image right and 90° straight up" in brief


def test_perspective_without_a_parallel_group_omits_the_angle_reading() -> None:
    result = perspective_test_result((perspective_test_family((1024.0, 256.0)),))

    assert "Parallel directions" not in measure_perspective(result, 512, 512).brief


def test_perspective_counts_short_edges_apart_from_unassigned() -> None:
    result = dataclasses.replace(
        perspective_test_result((perspective_test_family((1024.0, 256.0)),)),
        unassigned=5,
        too_short=12,
    )

    brief = measure_perspective(result, 512, 512).brief

    assert "5 edges fit no group, and 12 are too short to have a direction." in brief


def test_perspective_says_when_the_edge_limit_was_reached() -> None:
    """Eight edges in the group, five in none, and twelve too short: 25, against a limit of 25."""
    result = dataclasses.replace(
        perspective_test_result((perspective_test_family((1024.0, 256.0)),)),
        unassigned=5,
        too_short=12,
    )

    at_limit = measure_perspective(result, 512, 512, edge_limit=25).brief
    below_limit = measure_perspective(result, 512, 512, edge_limit=26).brief

    assert "The image reached the limit of 25 detected edges" in at_limit
    assert "limit" not in below_limit


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
    horizon = Horizon(
        left_y=0.25, right_y=0.75, vertical_family=0, source_families=(0,), assumption="a test"
    )
    result = dataclasses.replace(
        perspective_test_result((perspective_test_family((1024.0, 256.0)),)), horizon=horizon
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
        source_families=(0, 1, 2),
        vertical_family=None,
        farthest_point_diagonals=2.0,
        assumption="a test",
    )
    families = tuple(perspective_test_family((1024.0, 256.0)) for _ in range(3))
    result = perspective_test_result(families, camera)

    brief = measure_perspective(result, 512, 512).brief

    assert (
        "Camera estimate from 3 pairs among group 1 (red), group 2 (green) and group 3 (blue) "
        "that agree within 4%: field of view 60° across the width, assuming a test."
    ) in brief
    assert "The tilt is read from" not in brief


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
    result = dataclasses.replace(perspective_test_result(), unassigned=20, too_short=3)

    measurement = measure_perspective(result, 512, 512)

    assert measurement.brief == (
        "No group of straight edges converges or runs parallel: 20 edges fit no group, and 3 "
        "are too short to have a direction."
    )


def comparison_test_alignment(ambiguous: bool = False) -> Alignment:
    """A fitted alignment that leaves positions where they are."""
    return Alignment(
        transform=np.eye(3),
        fitted=True,
        matched=40,
        inliers=30,
        coverage=0.5,
        second_fit_inliers=20 if ambiguous else 0,
        ambiguous=ambiguous,
    )


def comparison_test_rows(offsets: list[float]) -> tuple[np.ndarray, np.ndarray, Pairing]:
    """Horizontal edges on rows 40, 80, and so on, each paired with one ``offsets`` below it."""
    rows = [40.0 * (index + 1) for index in range(len(offsets))]
    first = np.array([[0.0, row, 128.0, row] for row in rows])
    second = first + np.array([[0.0, offset, 0.0, offset] for offset in offsets])
    pairs = tuple(
        EdgePair(index, index, (0.0, offset), (0.0, 0.0)) for index, offset in enumerate(offsets)
    )
    return first, second, Pairing(pairs, (), (), (), ())


def test_comparison_brief_lists_largest_offsets() -> None:
    """Offsets of 2, 8, and 5 pixels down a 256-pixel frame are 0.01, 0.03, and 0.02."""
    first, second, pairing = comparison_test_rows([2.0, 8.0, 5.0])

    brief = measure_comparison(
        comparison_test_alignment(), pairing, first, second, (256, 256), (256, 256)
    ).brief

    assert "3 pairs of edges; unmatched 0 in the first image and 0 in the second." in brief
    assert "(0.00,0.31)-(0.50,0.31) offset (+0.00, +0.03)" in brief
    assert brief.index("+0.03") < brief.index("+0.02") < brief.index("+0.01")


def test_comparison_brief_reports_the_alignment_support() -> None:
    first, second, pairing = comparison_test_rows([2.0])

    brief = measure_comparison(
        comparison_test_alignment(), pairing, first, second, (256, 256), (256, 256)
    ).brief

    assert (
        "Aligned by a transform fitted to 30 of 40 matching features, which cover 50% of "
        "the first image."
    ) in brief


def test_comparison_offsets_scale_through_the_crop() -> None:
    """An offset of 8 pixels in a 256-pixel frame showing the lower half of the image: 0.02."""
    first, second, pairing = comparison_test_rows([8.0])

    brief = measure_comparison(
        comparison_test_alignment(),
        pairing,
        first,
        second,
        (256, 256),
        (256, 256),
        CropRegion(0.0, 0.5, 1.0, 1.0),
    ).brief

    assert "(0.00,0.58)-(0.50,0.58) offset (+0.00, +0.02)" in brief


def test_comparison_full_gives_the_second_edge_and_the_overrun() -> None:
    """The second edge lies 8 pixels lower and runs 32 pixels, a quarter of the first, further."""
    first = np.array([[0.0, 40.0, 128.0, 40.0]])
    second = np.array([[0.0, 48.0, 160.0, 48.0]])
    pairing = Pairing((EdgePair(0, 0, (0.0, 8.0), (0.0, 32.0)),), (), (), (), ())

    full = measure_comparison(
        comparison_test_alignment(), pairing, first, second, (256, 256), (256, 256)
    ).full

    assert "to (0.00,0.19)-(0.62,0.19) in the second image, ends +0.00 and +0.25" in full


def test_comparison_full_counts_broken_pieces_apart_from_unmatched() -> None:
    first = np.array([[0.0, 40.0, 128.0, 40.0], [0.0, 200.0, 64.0, 200.0]])
    second = np.array([[0.0, 40.0, 60.0, 40.0], [70.0, 40.0, 128.0, 40.0]])
    pairing = Pairing((EdgePair(0, 0, (0.0, 0.0), (0.0, -68.0)),), (1,), (), (), (1,))

    measurement = measure_comparison(
        identity_alignment((256, 256), (256, 256)), pairing, first, second, (256, 256), (256, 256)
    )

    assert "unmatched 1 in the first image and 0 in the second" in measurement.brief
    assert "not counted as unmatched: 0 in the first image and 1 in the second" in measurement.full
    assert "Longest unmatched in the first image: (0.00,0.78)-(0.25,0.78)." in measurement.full
    assert "Compared in one shared frame as asked" in measurement.brief


def test_comparison_fitted_response_says_offsets_include_parallax() -> None:
    first, second, pairing = comparison_test_rows([2.0])

    brief = measure_comparison(
        comparison_test_alignment(), pairing, first, second, (256, 256), (256, 256)
    ).brief

    assert "Offsets are (right, down) as fractions of the first image's width and height." in brief
    assert "what is left after the fitted transform" in brief
    assert "include the parallax of depth" in brief
    assert "need not be the same physical edge" in brief


def test_comparison_shared_frame_response_names_the_asserted_frame() -> None:
    first, second, pairing = comparison_test_rows([2.0])

    brief = measure_comparison(
        identity_alignment((256, 256), (256, 256)), pairing, first, second, (256, 256), (256, 256)
    ).brief

    assert "measured in the shared frame that was asked for" in brief
    assert "parallax" not in brief


def test_comparison_full_explains_the_ends() -> None:
    first, second, pairing = comparison_test_rows([2.0])

    measurement = measure_comparison(
        comparison_test_alignment(), pairing, first, second, (256, 256), (256, 256)
    )

    assert "in lengths of the first edge; a negative value means it stops short" in measurement.full
    assert "Ends give" not in measurement.brief


def test_comparison_names_the_image_that_reached_the_edge_limit() -> None:
    """Three edges in the first image against a limit of 3, and one in the second."""
    first, _, _ = comparison_test_rows([2.0, 8.0, 5.0])
    _, second, pairing = comparison_test_rows([2.0])

    brief = measure_comparison(
        comparison_test_alignment(), pairing, first, second, (256, 256), (256, 256), edge_limit=3
    ).brief

    assert "The first image reached the limit of 3 detected edges" in brief
    assert "an edge left unmatched in the other image may be one of them" in brief


def test_comparison_withheld_text_names_the_edge_limit_too() -> None:
    first, second, _ = comparison_test_rows([2.0])

    brief = measure_comparison(
        Withheld("a test"), None, first, second, (256, 256), (256, 256), edge_limit=1
    ).brief

    assert "Both images reached the limit of 1 detected edges" in brief


def test_comparison_outcome_has_no_numbers() -> None:
    """The outcome is what a comparison reports with measurements off."""
    outcomes = [
        comparison_outcome(comparison_test_alignment()),
        comparison_outcome(comparison_test_alignment(ambiguous=True)),
        comparison_outcome(identity_alignment((256, 256), (256, 256))),
        comparison_outcome(Withheld("views too different to align")),
    ]

    assert not any(character.isdigit() for outcome in outcomes for character in outcome)
    assert "second alignment is supported nearly as well" in outcomes[1]
    assert outcomes[3] == (
        "Alignment withheld: views too different to align. No edges were paired."
    )


def test_comparison_withheld_text_has_no_pairs() -> None:
    first, second, _ = comparison_test_rows([2.0])

    brief = measure_comparison(
        Withheld("views too different to align"), None, first, second, (256, 256), (256, 256)
    ).brief

    assert brief == (
        "Alignment withheld: views too different to align. No edges were paired. Straight "
        "edges found: 1 in the first image and 1 in the second."
    )


def test_comparison_flags_ambiguous_alignment() -> None:
    first, second, pairing = comparison_test_rows([2.0])

    brief = measure_comparison(
        comparison_test_alignment(ambiguous=True), pairing, first, second, (256, 256), (256, 256)
    ).brief

    assert "A second alignment fits 20 of the other features" in brief
    assert "the pairs assume the first" in brief


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
