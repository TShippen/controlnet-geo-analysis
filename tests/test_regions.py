"""Tests for crop regions and their mapping back to the full image."""

import pytest

from controlnet_mcp.regions import CropError, CropRegion


def test_from_list_rejects_wrong_length() -> None:
    with pytest.raises(CropError, match="four"):
        CropRegion.from_list([0.1, 0.2, 0.3])


def test_from_list_rejects_out_of_range() -> None:
    with pytest.raises(CropError, match="between 0 and 1"):
        CropRegion.from_list([0.1, 0.1, 1.2, 0.5])


def test_from_list_rejects_inverted() -> None:
    with pytest.raises(CropError, match="top-left"):
        CropRegion.from_list([0.6, 0.1, 0.2, 0.5])


def test_pixel_box_rounds_outward() -> None:
    assert CropRegion(0.25, 0.105, 0.5, 0.355).pixel_box(200, 100) == (50, 10, 100, 36)


def test_pixel_box_rejects_tiny_crop() -> None:
    with pytest.raises(CropError, match="at least 16"):
        CropRegion(0.0, 0.0, 0.1, 0.1).pixel_box(100, 100)


def test_snapped_matches_pixel_box() -> None:
    snapped = CropRegion(0.25, 0.105, 0.5, 0.355).snapped(200, 100)

    assert snapped == CropRegion(0.25, 0.1, 0.5, 0.36)


def test_to_full_maps_the_center() -> None:
    assert CropRegion(0.5, 0.25, 1.0, 0.75).to_full(0.5, 0.5) == (0.75, 0.5)


def test_to_local_inverts_to_full() -> None:
    region = CropRegion(0.2, 0.1, 0.6, 0.5)

    assert region.to_local(*region.to_full(0.25, 0.75)) == pytest.approx((0.25, 0.75))


def test_contains_is_inclusive() -> None:
    """The left edge x 0.2 is inside; x 0.61 is past the right edge at 0.6."""
    region = CropRegion(0.2, 0.1, 0.6, 0.5)

    assert region.contains(0.2, 0.5)
    assert not region.contains(0.61, 0.3)


def test_digest_differs_by_region() -> None:
    first = CropRegion.from_list([0.1, 0.1, 0.5, 0.5])

    assert first.digest() == CropRegion.from_list([0.1, 0.1, 0.5, 0.5]).digest()
    assert first.digest() != CropRegion.from_list([0.1, 0.1, 0.6, 0.5]).digest()
