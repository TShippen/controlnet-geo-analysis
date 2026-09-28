"""Tests for grouping straight segments by perspective.

Every scene is built from segments placed by hand in a 512x512 frame whose
center is (256, 256). Converging sets come from ``converging_test_segments``,
which puts eight segments on lines through one point.
"""

import math

import numpy as np
import pytest

from conftest import (
    LEFT_TEST_POINT,
    PERSPECTIVE_TEST_FRAME,
    RIGHT_TEST_POINT,
    camera_scene_test_segments,
    converging_test_segments,
    two_point_test_segments,
    vertical_test_segments,
)
from controlnet_mcp.evidence import Withheld
from controlnet_mcp.perspective import (
    CameraEstimate,
    Horizon,
    PerspectiveResult,
    analyze_perspective,
)


def analyze_test_scene(segments: np.ndarray, cropped: bool = False) -> PerspectiveResult:
    """Analyze segments placed in the square test frame."""
    return analyze_perspective(segments, PERSPECTIVE_TEST_FRAME, PERSPECTIVE_TEST_FRAME, cropped)


def horizontal_test_segments(pieces: list[tuple[float, float, float]]) -> np.ndarray:
    """Horizontal segments, each given as its row and the columns of its two ends.

    Five more at rows 140 to 300 make the set large enough to form a family.
    """
    others = [(140.0, 100.0, 160.0), (180.0, 90.0, 170.0), (220.0, 120.0, 220.0)]
    others += [(260.0, 60.0, 170.0), (300.0, 200.0, 320.0)]
    return np.array([[x0, y, x1, y] for y, x0, x1 in [*pieces, *others]])


def test_two_point_scene_finds_both_vanishing_points() -> None:
    result = analyze_test_scene(two_point_test_segments())

    points = sorted(family.vanishing_point for family in result.families)
    assert len(points) == 2
    assert points[0] == pytest.approx(LEFT_TEST_POINT, abs=1.0)
    assert points[1] == pytest.approx(RIGHT_TEST_POINT, abs=1.0)
    assert all(family.scatter_degrees < 0.1 for family in result.families)
    assert result.unassigned == 0


def test_straight_verticals_are_parallel() -> None:
    result = analyze_test_scene(vertical_test_segments())

    (family,) = result.families
    assert family.vanishing_point is None
    assert family.direction_degrees == pytest.approx(90.0)
    assert family.length_share == pytest.approx(1.0)


def test_unrelated_directions_form_no_family() -> None:
    """Twenty segments 9 degrees apart, each tangent to a circle of radius 150.

    Only two tangents of a circle pass through any one point, so no point has
    six of these segments aimed at it.
    """
    segments = []
    for index in range(20):
        angle = math.radians(9.0 * index)
        along = np.array([math.cos(angle), math.sin(angle)])
        middle = np.array([256.0, 256.0]) + 150.0 * np.array([-along[1], along[0]])
        segments.append([*(middle - 40.0 * along), *(middle + 40.0 * along)])

    result = analyze_test_scene(np.array(segments))

    assert result.families == ()
    assert result.unassigned == 20


def test_short_segments_are_ignored() -> None:
    """Segments 10 pixels long are under the 20 pixels a direction needs."""
    segments = converging_test_segments(RIGHT_TEST_POINT)
    middles = (segments[:, :2] + segments[:, 2:]) / 2.0
    halves = (segments[:, 2:] - segments[:, :2]) / 2.0
    halves = 5.0 * halves / np.linalg.norm(halves, axis=1, keepdims=True)
    short = np.hstack([middles - halves, middles + halves])

    result = analyze_test_scene(short)

    assert result.families == ()
    assert result.unassigned == 0
    assert result.too_short == 8


def test_at_most_three_families() -> None:
    """Four sets of eight, aimed at points far enough apart that no set supports another's."""
    segments = np.vstack(
        [
            converging_test_segments(point)
            for point in (RIGHT_TEST_POINT, LEFT_TEST_POINT, (256.0, -600.0), (-600.0, 1200.0))
        ]
    )

    result = analyze_test_scene(segments)

    assert len(result.families) == 3
    assert [len(family.segment_indices) for family in result.families] == [8, 8, 8]
    assert result.unassigned == 8


def test_edge_aimed_near_another_point_joins_the_first_group() -> None:
    """One edge of the set aimed at (1056, -144) points 1.4 degrees off (256, 656).

    From its anchor (480, 440) the first point lies along (576, -584) and the
    second along (-224, 216), which are 45.4 and 44.0 degrees below level.
    That is inside the 3 degrees that join an edge to a group, and the group
    around (256, 656) carries the most length, so it is found first and takes
    the edge.
    """
    segments = np.vstack(
        [
            converging_test_segments(point)
            for point in ((1056.0, -144.0), (-144.0, -144.0), (256.0, 656.0))
        ]
    )

    result = analyze_test_scene(segments)

    assert [len(family.segment_indices) for family in result.families] == [9, 8, 7]
    assert 7 in result.families[0].segment_indices


def test_shared_line_reports_pieces_and_gaps() -> None:
    """Three pieces on row 100 span columns 0 to 260, with gaps at 50 to 80 and 130 to 200."""
    pieces = [(100.0, 0.0, 50.0), (100.0, 80.0, 130.0), (100.0, 200.0, 260.0)]

    result = analyze_test_scene(horizontal_test_segments(pieces))

    (line,) = result.shared_lines
    assert line.segment_indices == (0, 1, 2)
    assert (line.start, line.end) == ((0.0, 100.0), (260.0, 100.0))
    assert line.gaps == pytest.approx([(50 / 260, 80 / 260), (130 / 260, 200 / 260)])


def test_offset_pieces_are_not_shared() -> None:
    """Rows 100 and 104 are 4 pixels apart, past the 2 pixels that join pieces."""
    pieces = [(100.0, 0.0, 50.0), (104.0, 80.0, 130.0)]

    result = analyze_test_scene(horizontal_test_segments(pieces))

    assert len(result.families) == 1
    assert result.shared_lines == ()


def test_camera_from_perpendicular_vanishing_points() -> None:
    """Focal length 400 across a width of 512 is a field of view of 2 * atan(256 / 400)."""
    camera = analyze_test_scene(camera_scene_test_segments()).camera

    assert isinstance(camera, CameraEstimate)
    assert camera.field_of_view_degrees == pytest.approx(65.2, abs=0.5)
    assert camera.pitch_degrees == pytest.approx(0.0, abs=0.5)
    assert camera.roll_degrees == pytest.approx(0.0, abs=0.5)


def test_camera_pitch_from_converging_verticals() -> None:
    """Verticals meeting below the image center mean the camera looks down.

    The verticals meet at offset (0, 400) from the center and a horizontal
    direction at offset (800, -400). Their product is -160000, a focal length
    of 400, and atan(400 / 400) is a pitch of 45 degrees.
    """
    below = (256.0, 656.0)
    side = (1056.0, -144.0)
    segments = np.vstack([converging_test_segments(side), converging_test_segments(below)])

    camera = analyze_test_scene(segments).camera

    assert isinstance(camera, CameraEstimate)
    assert camera.field_of_view_degrees == pytest.approx(65.2, abs=0.5)
    assert camera.pitch_degrees == pytest.approx(-45.0, abs=0.5)
    assert camera.roll_degrees == pytest.approx(0.0, abs=0.5)


def test_camera_from_three_perpendicular_points_counts_three_pairs() -> None:
    """The three vanishing points of a focal length 400 camera turned and pitched down.

    The offsets from the center are (1000, -400), (-320, -400), and (0, 400).
    Every pair has a product of -160000, so the three pairs agree exactly.
    """
    segments = np.vstack(
        [
            converging_test_segments(point)
            for point in ((1256.0, -144.0), (-64.0, -144.0), (256.0, 656.0))
        ]
    )

    camera = analyze_test_scene(segments).camera

    assert isinstance(camera, CameraEstimate)
    assert camera.pairs == 3
    assert camera.disagreement == pytest.approx(0.0, abs=0.001)
    assert camera.field_of_view_degrees == pytest.approx(65.2, abs=0.5)


def test_camera_withheld_when_parallel_verticals_meet_an_off_center_horizon() -> None:
    """Verticals stay parallel while the side points sit 100 pixels above the center.

    The side offsets (692.8, -100) and (-230.9, -100) still give a focal
    length, sqrt(150000), but a level camera centered on the image would put
    them on the row of the center.
    """
    segments = np.vstack(
        [
            converging_test_segments((948.8, 156.0)),
            converging_test_segments((25.1, 156.0)),
            vertical_test_segments(),
        ]
    )

    camera = analyze_test_scene(segments).camera

    assert isinstance(camera, Withheld)
    assert "straightened" in camera.reason


def test_camera_withheld_when_cropped() -> None:
    camera = analyze_test_scene(camera_scene_test_segments(), cropped=True).camera

    assert isinstance(camera, Withheld)
    assert "crop" in camera.reason


def test_camera_withheld_for_non_perpendicular_families() -> None:
    """Both points lie to the right of the center, so their offsets cannot be perpendicular."""
    segments = np.vstack(
        [converging_test_segments((700.0, 256.0)), converging_test_segments((900.0, 256.0))]
    )

    camera = analyze_test_scene(segments).camera

    assert isinstance(camera, Withheld)
    assert "not perpendicular" in camera.reason


def test_camera_withheld_with_one_family() -> None:
    camera = analyze_test_scene(converging_test_segments(RIGHT_TEST_POINT)).camera

    assert isinstance(camera, Withheld)
    assert "fewer than two" in camera.reason


def test_camera_withheld_when_pairs_disagree() -> None:
    """Offsets (700, 100), (-300, 100), and (0, -800) from the center.

    The first pair gives a focal length of sqrt(200000), 447, and each of the
    other two pairs gives sqrt(80000), 283.
    """
    segments = np.vstack(
        [
            converging_test_segments(point)
            for point in ((956.0, 356.0), (-44.0, 356.0), (256.0, -544.0))
        ]
    )

    camera = analyze_test_scene(segments).camera

    assert isinstance(camera, Withheld)
    assert "disagree" in camera.reason


def test_horizon_through_side_points_with_vertical_family() -> None:
    horizon = analyze_test_scene(camera_scene_test_segments()).horizon

    assert isinstance(horizon, Horizon)
    assert horizon.left_y == pytest.approx(0.50, abs=0.005)
    assert horizon.right_y == pytest.approx(0.50, abs=0.005)


def test_derived_values_record_the_families_they_came_from() -> None:
    """The verticals carry the most length and are family 0; the converging sets are 1 and 2."""
    result = analyze_test_scene(camera_scene_test_segments())

    assert isinstance(result.horizon, Horizon)
    assert isinstance(result.camera, CameraEstimate)
    assert result.horizon.vertical_family == 0
    assert result.horizon.source_families == (1, 2)
    assert result.camera.source_families == (1, 2)
    assert result.camera.vertical_family == 0


def test_camera_records_how_far_its_farthest_point_lies() -> None:
    """The right point is 692.8 pixels from the center and the diagonal is 724.1: 0.96."""
    camera = analyze_test_scene(camera_scene_test_segments()).camera

    assert isinstance(camera, CameraEstimate)
    assert camera.farthest_point_diagonals == pytest.approx(0.957, abs=0.005)


def leaning_test_segments(lean_degrees: float) -> np.ndarray:
    """Eight segments 100 pixels long, at the columns ``vertical_test_segments`` uses.

    Each leans ``lean_degrees`` to the right at the top instead of running
    dead vertical, so the family they form reports that lean as its
    direction. At ``lean_degrees`` of 0 this is ``vertical_test_segments``.
    """
    along = math.radians(90.0 - lean_degrees)
    unit = np.array([math.cos(along), -math.sin(along)])
    segments = []
    for x in np.arange(60.0, 481.0, 60.0):
        center = np.array([x, 250.0])
        bottom, top = center - 50.0 * unit, center + 50.0 * unit
        segments.append([bottom[0], bottom[1], top[0], top[1]])
    return np.array(segments)


def test_horizon_through_one_point_follows_leaning_parallel_verticals() -> None:
    """Verticals leaning 5 degrees tilt the horizon by that much through the one open point."""
    segments = np.vstack([converging_test_segments(RIGHT_TEST_POINT), leaning_test_segments(5.0)])

    horizon = analyze_test_scene(segments).horizon

    assert isinstance(horizon, Horizon)
    assert horizon.left_y == pytest.approx(0.338, abs=0.005)
    assert horizon.right_y == pytest.approx(0.425, abs=0.005)
    assert "rolled" not in horizon.assumption


def test_horizon_through_one_point_is_level_under_upright_verticals() -> None:
    segments = np.vstack([converging_test_segments(RIGHT_TEST_POINT), vertical_test_segments()])

    horizon = analyze_test_scene(segments).horizon

    assert isinstance(horizon, Horizon)
    assert horizon.left_y == pytest.approx(0.50, abs=0.005)
    assert horizon.right_y == pytest.approx(0.50, abs=0.005)


def test_horizon_through_one_point_follows_converging_verticals() -> None:
    """The verticals' point sits at offset (200, 2268.5) from the center.

    Perpendicular to that line is a slope of -200 / 2268.5, through the one
    other point at (948.8, 185.5).
    """
    segments = np.vstack(
        [
            converging_test_segments((456.0, 2524.5)),
            converging_test_segments((948.8, 185.5)),
        ]
    )

    horizon = analyze_test_scene(segments).horizon

    assert isinstance(horizon, Horizon)
    assert horizon.left_y == pytest.approx(0.526, abs=0.005)
    assert horizon.right_y == pytest.approx(0.438, abs=0.005)
    assert "optical center" in horizon.assumption


def test_horizon_withheld_for_a_crop_with_converging_verticals() -> None:
    segments = np.vstack(
        [
            converging_test_segments((456.0, 2524.5)),
            converging_test_segments((948.8, 185.5)),
        ]
    )

    horizon = analyze_test_scene(segments, cropped=True).horizon

    assert isinstance(horizon, Withheld)
    assert "crop" in horizon.reason


def test_horizon_withheld_without_near_vertical_family() -> None:
    horizon = analyze_test_scene(two_point_test_segments()).horizon

    assert isinstance(horizon, Withheld)
    assert "vertical" in horizon.reason


def receding_road_test_segments() -> np.ndarray:
    """A camera pitched 10 degrees down along a road.

    Sixteen edges recede along the road, converging just above the image
    center but still inside the frame, at (256, 185.5). Eight edges are the
    scene's verticals, converging far below the frame at (256, 2524.5). The
    offsets from the center, -70.5 and 2268.5, have a product of 400
    squared, the focal length of the scene. Both groups' vanishing points
    sit on the center column, so both read as dead vertical by bearing
    alone; only the one outside the frame is the true verticals.
    """
    ahead = np.vstack([converging_test_segments((256.0, 185.5))] * 2)
    verticals = converging_test_segments((256.0, 2524.5))
    return np.vstack([ahead, verticals])


def test_receding_group_is_not_taken_for_the_verticals() -> None:
    """The group receding along the road is dead vertical by bearing but meets inside the frame."""
    result = analyze_test_scene(receding_road_test_segments())
    camera = result.camera

    assert isinstance(camera, CameraEstimate)
    assert camera.vertical_family is not None
    vertical = result.families[camera.vertical_family]
    assert vertical.vanishing_point == pytest.approx((256.0, 2524.5), abs=1.0)
    assert camera.pitch_degrees == pytest.approx(-10.0, abs=0.5)
    assert camera.field_of_view_degrees == pytest.approx(65.2, abs=0.5)


def test_group_meeting_inside_the_frame_gives_no_tilt() -> None:
    """Neither group qualifies for the verticals: one meets inside the frame, the other is level."""
    segments = np.vstack(
        [converging_test_segments((256.0, 185.5)), converging_test_segments(RIGHT_TEST_POINT)]
    )

    result = analyze_test_scene(segments)

    assert isinstance(result.horizon, Withheld)
    assert "outside the frame" in result.horizon.reason
    if isinstance(result.camera, CameraEstimate):
        assert result.camera.pitch_degrees is None
        assert result.camera.roll_degrees is None


def test_parallel_verticals_win_over_a_converging_group() -> None:
    """A parallel group and a converging group both read as dead vertical by bearing.

    The point at (256, -50) meets outside the frame, but a group held
    exactly parallel is still taken for the verticals over one that only
    converges. The converging edges here run more than 3 degrees off
    vertical, so the two families stay distinct.
    """
    result = analyze_test_scene(
        np.vstack([vertical_test_segments(), converging_test_segments((256.0, -50.0))])
    )

    parallel_index = next(
        index for index, family in enumerate(result.families) if family.vanishing_point is None
    )
    assert isinstance(result.horizon, Horizon)
    assert result.horizon.vertical_family == parallel_index


def test_farthest_point_wins_among_converging_groups() -> None:
    """Two converging groups both read as dead vertical by bearing and meet outside the frame.

    A camera pitched 30 degrees down, focal length 600. The receding group
    meets at (256, -90.4), 346.4 pixels above the center, and the verticals
    meet at (256, 1295.2), 1039.2 pixels below; 346.4 times 1039.2 is 600
    squared. The receding group carries sixteen edges against the
    verticals' eight, so length alone would not decide it: the farther
    point wins because it lies farther from the image center.
    """
    ahead = np.vstack([converging_test_segments((256.0, -90.4))] * 2)
    verticals = converging_test_segments((256.0, 1295.2))

    result = analyze_test_scene(np.vstack([ahead, verticals]))
    camera = result.camera

    assert isinstance(camera, CameraEstimate)
    assert camera.vertical_family is not None
    vertical = result.families[camera.vertical_family]
    assert vertical.vanishing_point == pytest.approx((256.0, 1295.2), abs=1.0)
    assert camera.pitch_degrees == pytest.approx(-30.0, abs=0.5)
    assert camera.field_of_view_degrees == pytest.approx(46.2, abs=0.5)


def test_camera_assumption_names_the_verticals_when_tilt_is_given() -> None:
    camera = analyze_test_scene(receding_road_test_segments()).camera

    assert isinstance(camera, CameraEstimate)
    assert "vertical in the scene" in camera.assumption
    assert "less than 45" in camera.assumption


def test_camera_assumption_without_verticals_leaves_them_out() -> None:
    camera = analyze_test_scene(two_point_test_segments()).camera

    assert isinstance(camera, CameraEstimate)
    assert camera.pitch_degrees is None
    assert "vertical in the scene" not in camera.assumption
