"""Tests for reading values off decoded depth and normal maps.

Every map is built in the test, and every expected value is counted by hand
from it. A sample reads the 5-pixel window centered on pixel
``floor(position * size)``.
"""

import numpy as np
import pytest

from controlnet_mcp.processors import read_depth_values, read_normal_values
from controlnet_mcp.regions import FULL_IMAGE, CropRegion
from controlnet_mcp.sampling import (
    SamplingError,
    ValueMap,
    parse_line,
    parse_points,
    sample_line,
    sample_points,
)

FACING_CAMERA = (128, 128, 255)
TURNED_LEFT = (218, 128, 218)


def depth_test_map(levels: np.ndarray) -> ValueMap:
    """Decode a 2D array of gray levels as a rendered depth map."""
    gray = levels.astype(np.uint8)
    return read_depth_values(np.stack([gray, gray, gray], axis=2))


def halves_test_levels(left: int, right: int, size: int = 64) -> np.ndarray:
    """A square of gray levels whose left and right halves each hold one level."""
    levels = np.full((size, size), left)
    levels[:, size // 2 :] = right
    return levels


def test_depth_reader_levels_and_beyond_range() -> None:
    """Levels at or below 2 are beyond the depth range; 3 is the first ordinary level."""
    value_map = depth_test_map(np.array([[0, 2, 3, 200]]))

    assert value_map.values[0, :, 0].tolist() == [0, 2, 3, 200]
    assert value_map.beyond_range[0].tolist() == [True, True, False, False]


def test_normals_reader_decodes_camera_axes() -> None:
    """Red 218 decodes to 0.71 and faces left, so right is -0.71; green 128 is level."""
    value_map = read_normal_values(np.array([[TURNED_LEFT]], dtype=np.uint8))

    assert value_map.values[0, 0].tolist() == pytest.approx([-0.71, 0.0, 0.71], abs=0.01)


def test_flat_region_sample_has_zero_spread() -> None:
    value_map = depth_test_map(np.full((64, 64), 120))

    (sample,) = sample_points(value_map, [(0.5, 0.5)], FULL_IMAGE)

    assert sample.value == [120]
    assert sample.spread == 0
    assert sample.on_boundary is False
    assert sample.beyond_range is False


def test_sample_on_step_flags_boundary() -> None:
    """Pixel 32 has a window over columns 30 to 34, which holds both 60 and 200."""
    value_map = depth_test_map(halves_test_levels(60, 200))

    (sample,) = sample_points(value_map, [(0.5, 0.5)], FULL_IMAGE)

    assert sample.spread == 140
    assert sample.on_boundary is True


def test_sample_in_clipped_black_has_no_value() -> None:
    value_map = depth_test_map(np.zeros((64, 64)))

    (sample,) = sample_points(value_map, [(0.5, 0.5)], FULL_IMAGE)

    assert sample.value is None
    assert sample.beyond_range is True


def test_points_map_through_crop() -> None:
    """In a crop of the right half, x 0.6 is 0.2 across the map and x 0.9 is 0.8 across."""
    value_map = depth_test_map(halves_test_levels(50, 150))
    region = CropRegion(0.5, 0.0, 1.0, 1.0)

    left, right = sample_points(value_map, [(0.6, 0.5), (0.9, 0.5)], region)

    assert left.value == [50]
    assert right.value == [150]


def test_point_outside_crop_is_rejected() -> None:
    value_map = depth_test_map(halves_test_levels(50, 150))

    with pytest.raises(SamplingError):
        sample_points(value_map, [(0.2, 0.5)], CropRegion(0.5, 0.0, 1.0, 1.0))


def test_line_samples_are_even_and_inclusive() -> None:
    value_map = depth_test_map(np.full((64, 64), 120))

    samples, _ = sample_line(value_map, (0.1, 0.5), (0.9, 0.5), 5, FULL_IMAGE)

    assert [sample.x for sample in samples] == pytest.approx([0.1, 0.3, 0.5, 0.7, 0.9])
    assert [sample.y for sample in samples] == pytest.approx([0.5] * 5)


def test_line_across_three_bands_reports_two_changes() -> None:
    """Bands of 200, 150, and 100 meet at columns 85 and 171 of 256.

    Sample i sits at column floor(256 * i / 31). Sample 10 is column 82 and
    sample 11 is column 90, on either side of 85; sample 20 is column 165 and
    sample 21 is column 173, on either side of 171.
    """
    levels = np.full((64, 256), 200)
    levels[:, 85:171] = 150
    levels[:, 171:] = 100

    _, changes = sample_line(depth_test_map(levels), (0.0, 0.5), (1.0, 0.5), 32, FULL_IMAGE)

    assert [change.after_sample for change in changes] == [10, 20]
    assert [change.size for change in changes] == [-50, -50]


def test_gentle_ramp_reports_no_change() -> None:
    """A ramp of 10 levels over the map moves under one level between samples."""
    levels = np.tile(np.linspace(100, 110, 256), (64, 1))

    _, changes = sample_line(depth_test_map(levels), (0.0, 0.5), (1.0, 0.5), 32, FULL_IMAGE)

    assert changes == []


def gradual_step_test_levels() -> np.ndarray:
    """Level 100 rising to 156 by 7 levels a column over columns 124 to 131 of 256."""
    levels = np.full((64, 256), 100)
    levels[:, 124:132] = 100 + 7 * np.arange(1, 9)
    levels[:, 132:] = 156
    return levels


def test_sample_on_gradual_step_is_not_flagged() -> None:
    """Pixel 128 has a window over columns 126 to 130: levels 121 to 149, a spread of 28."""
    value_map = depth_test_map(gradual_step_test_levels())

    (sample,) = sample_points(value_map, [(0.5, 0.5)], FULL_IMAGE)

    assert sample.spread == 28
    assert sample.on_boundary is False


def test_line_across_gradual_step_reports_the_change() -> None:
    """Sample 15 is column 123, before the rise, and sample 16 is column 132, after it."""
    value_map = depth_test_map(gradual_step_test_levels())

    _, changes = sample_line(value_map, (0.0, 0.5), (1.0, 0.5), 32, FULL_IMAGE)

    assert [(change.after_sample, change.size) for change in changes] == [(15, 56)]


def test_change_into_beyond_range_has_no_size() -> None:
    value_map = depth_test_map(halves_test_levels(150, 0))

    _, changes = sample_line(value_map, (0.0, 0.5), (1.0, 0.5), 32, FULL_IMAGE)

    assert [change.size for change in changes] == [None]


def test_normals_change_measured_in_degrees() -> None:
    """A face toward the camera beside one turned 45 degrees left."""
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[:, :32] = FACING_CAMERA
    rgb[:, 32:] = TURNED_LEFT

    _, changes = sample_line(read_normal_values(rgb), (0.0, 0.5), (1.0, 0.5), 32, FULL_IMAGE)

    assert [change.size for change in changes] == [pytest.approx(45, abs=1)]


def test_line_count_out_of_range_is_rejected() -> None:
    value_map = depth_test_map(np.full((64, 64), 120))

    with pytest.raises(SamplingError):
        sample_line(value_map, (0.0, 0.5), (1.0, 0.5), 1, FULL_IMAGE)


def test_parse_points_rejects_a_value_outside_the_image() -> None:
    with pytest.raises(SamplingError):
        parse_points([[0.5, 1.2]])


def test_parse_points_rejects_too_many() -> None:
    with pytest.raises(SamplingError):
        parse_points([[0.5, 0.5]] * 65)


def test_parse_line_rejects_wrong_length() -> None:
    with pytest.raises(SamplingError):
        parse_line([0.1, 0.2, 0.3])
