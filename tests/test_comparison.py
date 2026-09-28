"""Tests for aligning two images and pairing their straight edges.

Textures are 256x256 images of random gray blocks 16 pixels on a side, built
here from a seeded generator. Segments are placed by hand, and every expected
offset is counted from them.
"""

import cv2
import numpy as np
import pytest

from controlnet_mcp.comparison import (
    Alignment,
    ComparisonError,
    align_images,
    identity_alignment,
    map_segments,
    pair_edges,
    render_comparison,
)
from controlnet_mcp.evidence import Withheld

FRAME = (256, 256)
IDENTITY = np.eye(3)
CORNERS = np.array([[0.0, 0.0, 256.0, 0.0], [256.0, 256.0, 0.0, 256.0]])


def texture_test_image(seed: int) -> np.ndarray:
    """A 256x256 grayscale image of random blocks 16 pixels on a side."""
    blocks = np.random.default_rng(seed).integers(0, 256, size=(16, 16), dtype=np.uint8)
    return np.kron(blocks, np.ones((16, 16), dtype=np.uint8))


def shifted_right_test_image(image: np.ndarray, pixels: int) -> np.ndarray:
    """The image moved right by ``pixels``, its left margin filled with mid gray."""
    shifted = np.full_like(image, 128)
    shifted[:, pixels:] = image[:, :-pixels]
    return shifted


def test_opencv_provides_sift_and_magsac() -> None:
    """Pins the two parts of the library that alignment is built on."""
    assert callable(cv2.SIFT.create)
    assert isinstance(cv2.USAC_MAGSAC, int)


def test_identical_images_align_to_identity() -> None:
    texture = texture_test_image(1)

    alignment = align_images(texture, texture)

    assert isinstance(alignment, Alignment)
    assert map_segments(CORNERS, alignment.transform) == pytest.approx(CORNERS, abs=0.5)
    assert alignment.ambiguous is False


def test_shifted_image_recovers_the_shift() -> None:
    """The second image shows the texture 10 pixels to the right.

    The transform takes second-image positions to first-image positions, so
    it moves every corner 10 pixels to the left.
    """
    texture = texture_test_image(1)

    alignment = align_images(texture, shifted_right_test_image(texture, 10))

    assert isinstance(alignment, Alignment)
    moved = map_segments(CORNERS, alignment.transform) - CORNERS
    assert moved[:, [0, 2]] == pytest.approx(np.full((2, 2), -10.0), abs=0.5)
    assert moved[:, [1, 3]] == pytest.approx(np.zeros((2, 2)), abs=0.5)


def test_fitted_alignment_reports_its_support() -> None:
    texture = texture_test_image(1)

    alignment = align_images(texture, texture)

    assert isinstance(alignment, Alignment)
    assert alignment.fitted is True
    assert alignment.inliers >= 12
    assert alignment.inliers <= alignment.matched
    assert alignment.coverage >= 0.125


def test_unrelated_images_are_withheld() -> None:
    alignment = align_images(texture_test_image(1), texture_test_image(2))

    assert isinstance(alignment, Withheld)
    assert "too different" in alignment.reason


def test_plain_images_are_withheld() -> None:
    plain = np.full((256, 256), 128, dtype=np.uint8)

    assert isinstance(align_images(plain, plain), Withheld)


def test_repeating_pattern_is_never_an_unflagged_alignment() -> None:
    """One 32-pixel tile repeated 8 times each way, against itself moved right by one tile.

    Every tile looks like every other, so the features inside the pattern
    match nothing in particular and only a few near the borders match. Those
    are found at several orientations each, and counted once each they are
    too few to fit a transform.
    """
    tile = np.random.default_rng(3).integers(0, 256, size=(4, 4), dtype=np.uint8)
    pattern = np.kron(np.tile(tile, (8, 8)), np.ones((8, 8), dtype=np.uint8))

    alignment = align_images(pattern, shifted_right_test_image(pattern, 32))

    assert isinstance(alignment, Withheld) or alignment.ambiguous


def test_two_planes_moving_apart_are_flagged_ambiguous() -> None:
    """The left half of the texture moved right by 10 pixels and the right half by 30.

    Each half supports its own transform, and the smaller support is more
    than half of the larger.
    """
    texture = texture_test_image(1)
    second = np.full_like(texture, 128)
    second[:, 10:128] = texture[:, :118]
    second[:, 158:] = texture[:, 128:226]

    alignment = align_images(texture, second)

    assert isinstance(alignment, Alignment)
    assert alignment.ambiguous is True
    assert alignment.second_fit_inliers >= 0.5 * alignment.inliers


def test_identity_scales_the_second_frame_onto_the_first() -> None:
    """A second frame half the size of the first maps its corner (128, 64) to (256, 128)."""
    alignment = identity_alignment((256, 128), (128, 64), (256, 128), (128, 64))

    assert alignment.fitted is False
    mapped = map_segments(np.array([[0.0, 0.0, 128.0, 64.0]]), alignment.transform)
    assert mapped.tolist() == [[0.0, 0.0, 256.0, 128.0]]


def test_identity_rejects_different_aspect_ratios() -> None:
    with pytest.raises(ComparisonError):
        identity_alignment((64, 32), (16, 16), (64, 32), (16, 16))


def test_identity_rejects_proportions_the_detection_sizes_hide() -> None:
    """1000x750 and 1000x720 both resize to 704x512, which would hide their difference."""
    with pytest.raises(ComparisonError):
        identity_alignment((704, 512), (704, 512), (1000, 750), (1000, 720))


def test_identity_accepts_one_frame_at_two_sizes() -> None:
    """1000x750 and 2000x1500 are the same frame at two resolutions, so they share it."""
    alignment = identity_alignment((704, 512), (704, 512), (1000, 750), (2000, 1500))

    assert alignment.transform == pytest.approx(IDENTITY)


def test_identity_accepts_a_pixel_of_difference() -> None:
    """A crop fitted to whole pixels can leave sources a pixel apart, still counted as one frame."""
    identity_alignment((704, 512), (704, 512), (1000, 750), (1001, 750))


def test_pairing_reports_a_known_offset() -> None:
    first = np.array([[0.0, 100.0, 200.0, 100.0]])
    second = np.array([[0.0, 104.0, 200.0, 104.0]])

    pairing = pair_edges(first, second, IDENTITY, FRAME)

    (pair,) = pairing.pairs
    assert (pair.first_index, pair.second_index) == (0, 0)
    assert pair.offset == pytest.approx((0.0, 4.0))
    assert pair.end_offsets == pytest.approx((0.0, 0.0))


def test_pairing_reports_how_far_the_second_edge_overruns() -> None:
    """The second edge starts 10 pixels late and runs 20 pixels past the end."""
    first = np.array([[0.0, 100.0, 200.0, 100.0]])
    second = np.array([[10.0, 100.0, 220.0, 100.0]])

    (pair,) = pair_edges(first, second, IDENTITY, FRAME).pairs

    assert pair.end_offsets == pytest.approx((-10.0, 20.0))


def test_pairing_maps_the_second_image_through_the_transform() -> None:
    """A second frame of half the size: its edge on row 52 lands on row 104 of the first."""
    first = np.array([[0.0, 100.0, 200.0, 100.0]])
    second = np.array([[0.0, 52.0, 100.0, 52.0]])
    transform = identity_alignment((256, 256), (128, 128), (256, 256), (128, 128)).transform

    (pair,) = pair_edges(first, second, transform, FRAME).pairs

    assert pair.offset == pytest.approx((0.0, 4.0))


def test_pairing_is_mutual_best() -> None:
    """The second edge on row 103 is 3 from row 100 and 5 from row 108."""
    first = np.array([[0.0, 100.0, 200.0, 100.0], [0.0, 108.0, 200.0, 108.0]])
    second = np.array([[0.0, 103.0, 200.0, 103.0]])

    pairing = pair_edges(first, second, IDENTITY, FRAME)

    assert [(pair.first_index, pair.second_index) for pair in pairing.pairs] == [(0, 0)]
    assert pairing.unmatched_first == (1,)
    assert pairing.unmatched_second == ()


def test_pairing_is_symmetric_under_swap() -> None:
    first = np.array([[0.0, 100.0, 200.0, 100.0], [0.0, 108.0, 200.0, 108.0]])
    second = np.array([[0.0, 103.0, 200.0, 103.0]])

    pairing = pair_edges(second, first, IDENTITY, FRAME)

    assert [(pair.first_index, pair.second_index) for pair in pairing.pairs] == [(0, 0)]
    assert pairing.unmatched_first == ()
    assert pairing.unmatched_second == (1,)


def test_edge_in_line_with_a_pair_but_not_covered_is_unmatched() -> None:
    """The unpaired edge lies on the paired edge's line, but the second image has no edge there."""
    first = np.array([[0.0, 100.0, 80.0, 100.0], [150.0, 100.0, 230.0, 100.0]])
    second = np.array([[0.0, 100.0, 80.0, 100.0]])

    pairing = pair_edges(first, second, IDENTITY, FRAME)

    assert [(pair.first_index, pair.second_index) for pair in pairing.pairs] == [(0, 0)]
    assert pairing.unmatched_first == (1,)
    assert pairing.on_matched_line_first == ()


def test_broken_edge_pieces_sit_on_the_matched_line() -> None:
    """The second image's one edge covers both first-image pieces along their shared line."""
    first = np.array([[0.0, 100.0, 80.0, 100.0], [90.0, 100.0, 230.0, 100.0]])
    second = np.array([[0.0, 102.0, 230.0, 102.0]])

    pairing = pair_edges(first, second, IDENTITY, FRAME)

    assert [(pair.first_index, pair.second_index) for pair in pairing.pairs] == [(0, 0)]
    assert pairing.on_matched_line_first == (1,)
    assert pairing.unmatched_first == ()


def test_piece_less_than_half_covered_is_unmatched() -> None:
    """The second image's edge covers only 30 of the second first-image piece's 80 pixels."""
    first = np.array([[0.0, 100.0, 80.0, 100.0], [150.0, 100.0, 230.0, 100.0]])
    second = np.array([[0.0, 100.0, 180.0, 100.0]])

    pairing = pair_edges(first, second, IDENTITY, FRAME)

    assert pairing.unmatched_first == (1,)


def test_piece_half_covered_counts_as_a_piece() -> None:
    """The second image's edge covers 40 of the second first-image piece's 80 pixels."""
    first = np.array([[0.0, 100.0, 80.0, 100.0], [150.0, 100.0, 230.0, 100.0]])
    second = np.array([[0.0, 100.0, 190.0, 100.0]])

    pairing = pair_edges(first, second, IDENTITY, FRAME)

    assert pairing.on_matched_line_first == (1,)


def test_pieces_are_the_same_in_either_order() -> None:
    """The broken-edge-pieces scene with the images swapped finds the same piece."""
    first = np.array([[0.0, 102.0, 230.0, 102.0]])
    second = np.array([[0.0, 100.0, 80.0, 100.0], [90.0, 100.0, 230.0, 100.0]])

    pairing = pair_edges(first, second, IDENTITY, FRAME)

    assert pairing.on_matched_line_second == (1,)
    assert pairing.unmatched_second == ()


def test_no_pair_beyond_angle_tolerance() -> None:
    """The second edge is turned 10 degrees about the midpoint (100, 100) of the first."""
    first = np.array([[0.0, 100.0, 200.0, 100.0]])
    reach = 100.0 * np.array([np.cos(np.radians(10.0)), np.sin(np.radians(10.0))])
    second = np.array([[100.0 - reach[0], 100.0 - reach[1], 100.0 + reach[0], 100.0 + reach[1]]])

    pairing = pair_edges(first, second, IDENTITY, FRAME)

    assert pairing.pairs == ()
    assert pairing.unmatched_first == (0,)
    assert pairing.unmatched_second == (0,)


def test_no_pair_beyond_distance_tolerance() -> None:
    """Rows 100 and 114 are 14 pixels apart, past 5% of a 256-pixel frame, 12.8."""
    first = np.array([[0.0, 100.0, 200.0, 100.0]])
    second = np.array([[0.0, 114.0, 200.0, 114.0]])

    assert pair_edges(first, second, IDENTITY, FRAME).pairs == ()


def test_no_pair_between_edges_that_do_not_run_alongside() -> None:
    """Two edges on one row, one ending at column 80 and the other starting at column 120."""
    first = np.array([[0.0, 100.0, 80.0, 100.0]])
    second = np.array([[120.0, 100.0, 200.0, 100.0]])

    assert pair_edges(first, second, IDENTITY, FRAME).pairs == ()


def test_pairing_with_no_edges_in_one_image() -> None:
    first = np.array([[0.0, 100.0, 200.0, 100.0]])

    pairing = pair_edges(first, np.zeros((0, 4)), IDENTITY, FRAME)

    assert pairing.pairs == ()
    assert pairing.unmatched_first == (0,)


def test_render_draws_both_edge_sets_and_the_offset() -> None:
    """Cyan on row 100, magenta on row 110, and yellow joining their midpoints at column 100."""
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    first = np.array([[0.0, 100.0, 200.0, 100.0]])
    second = np.array([[0.0, 110.0, 200.0, 110.0]])
    alignment = identity_alignment(FRAME, FRAME, FRAME, FRAME)
    pairing = pair_edges(first, second, alignment.transform, FRAME)

    canvas = render_comparison(image, image, first, second, alignment, pairing)

    assert canvas.shape == (256, 256, 3)
    assert canvas[100, 50].tolist() == [0, 255, 255]
    assert canvas[110, 50].tolist() == [255, 0, 255]
    assert canvas[105, 100].tolist() == [255, 255, 0]


def test_render_without_alignment_is_two_panels() -> None:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    first = np.array([[0.0, 100.0, 200.0, 100.0]])
    second = np.array([[0.0, 110.0, 200.0, 110.0]])

    canvas = render_comparison(image, image, first, second, Withheld("a test"), None)

    assert canvas.shape == (256, 512, 3)
    assert canvas[100, 50].tolist() == [0, 255, 255]
    assert canvas[110, 256 + 50].tolist() == [255, 0, 255]
