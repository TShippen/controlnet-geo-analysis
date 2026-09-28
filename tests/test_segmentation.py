"""Tests for region prompts, overlay rendering, and the prompted segmenter."""

from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pytest
import torch
from controlnet_aux.segment_anything.build_sam import sam_model_registry
from PIL import Image

from controlnet_mcp.checkpoints import MOBILE_SAM_CHECKPOINT, CheckpointSpec, checkpoint_path
from controlnet_mcp.segmentation import (
    PromptedSegmenter,
    PromptError,
    RegionPrompt,
    choose_mask,
    prediction_inputs,
    render_region_overlay,
)


def test_prompt_requires_box_or_point() -> None:
    with pytest.raises(PromptError):
        RegionPrompt.from_lists(None, None)


def test_prompt_rejects_wrong_lengths() -> None:
    with pytest.raises(PromptError):
        RegionPrompt.from_lists([0.1, 0.2, 0.3], None)
    with pytest.raises(PromptError):
        RegionPrompt.from_lists(None, [0.5])


def test_prompt_rejects_out_of_range() -> None:
    with pytest.raises(PromptError):
        RegionPrompt.from_lists(None, [1.2, 0.5])


def test_prompt_rejects_inverted_box() -> None:
    with pytest.raises(PromptError):
        RegionPrompt.from_lists([0.6, 0.1, 0.4, 0.9], None)


def test_prompt_accepts_box_and_point_together() -> None:
    prompt = RegionPrompt.from_lists([0.1, 0.2, 0.8, 0.9], [0.5, 0.5])

    assert prompt.box == (0.1, 0.2, 0.8, 0.9)
    assert prompt.point == (0.5, 0.5)


def test_prompt_digest_is_stable_and_order_sensitive() -> None:
    first = RegionPrompt.from_lists(None, [0.2, 0.7]).digest()
    same = RegionPrompt.from_lists(None, [0.2, 0.7]).digest()
    swapped = RegionPrompt.from_lists(None, [0.7, 0.2]).digest()

    assert first == same
    assert first != swapped
    assert len(first) == 8


def test_prompt_digest_ignores_float_noise() -> None:
    plain = RegionPrompt.from_lists(None, [0.0, 0.3]).digest()
    negative_zero = RegionPrompt.from_lists(None, [-0.0, 0.3]).digest()
    noisy = RegionPrompt.from_lists(None, [0.0, 0.1 + 0.2]).digest()

    assert negative_zero == plain
    assert noisy == plain


def test_prompt_rejects_malformed_exclude_point() -> None:
    with pytest.raises(PromptError):
        RegionPrompt.from_lists(None, [0.5, 0.5], exclude=[[0.1, 0.2, 0.3]])


def test_prompt_rejects_extent_with_a_box() -> None:
    with pytest.raises(PromptError):
        RegionPrompt.from_lists([0.1, 0.1, 0.9, 0.9], None, extent="largest")


def test_prompt_rejects_extent_with_exclude_points() -> None:
    with pytest.raises(PromptError):
        RegionPrompt.from_lists(None, [0.5, 0.5], exclude=[[0.9, 0.5]], extent="smallest")


def test_prompt_digest_differs_by_exclude_and_extent() -> None:
    plain = RegionPrompt.from_lists(None, [0.5, 0.5]).digest()
    excluded = RegionPrompt.from_lists(None, [0.5, 0.5], exclude=[[0.9, 0.5]]).digest()
    largest = RegionPrompt.from_lists(None, [0.5, 0.5], extent="largest").digest()

    assert len({plain, excluded, largest}) == 3


def test_lone_point_asks_for_several_masks() -> None:
    inputs = prediction_inputs(RegionPrompt.from_lists(None, [0.5, 0.25]), 200, 100)

    assert inputs.multimask is True
    assert inputs.point_coords is not None
    assert inputs.point_coords.tolist() == [[100.0, 25.0]]
    assert inputs.point_labels is not None
    assert inputs.point_labels.tolist() == [1]
    assert inputs.box is None


def test_box_asks_for_one_mask() -> None:
    inputs = prediction_inputs(RegionPrompt.from_lists([0.1, 0.2, 0.5, 0.6], None), 200, 100)

    assert inputs.multimask is False
    assert inputs.box is not None
    assert inputs.box.tolist() == [20.0, 20.0, 100.0, 60.0]
    assert inputs.point_coords is None


def test_exclude_points_are_labelled_background() -> None:
    prompt = RegionPrompt.from_lists(None, [0.5, 0.5], exclude=[[0.9, 0.5], [0.1, 0.5]])

    inputs = prediction_inputs(prompt, 200, 100)

    assert inputs.multimask is False
    assert inputs.point_coords is not None
    assert inputs.point_coords.tolist() == [[100.0, 50.0], [180.0, 50.0], [20.0, 50.0]]
    assert inputs.point_labels is not None
    assert inputs.point_labels.tolist() == [1, 0, 0]


def candidate_masks() -> tuple[np.ndarray, np.ndarray]:
    """Three 4x4 candidates covering 4, 16, and 8 pixels, scored so the middle area scores best."""
    masks = np.zeros((3, 4, 4), dtype=bool)
    masks[0, :1, :] = True
    masks[1, :, :] = True
    masks[2, :2, :] = True
    return masks, np.array([0.5, 0.7, 0.9])


def test_choose_best_takes_the_highest_score() -> None:
    masks, scores = candidate_masks()

    assert choose_mask(masks, scores, "best") == 2


def test_choose_largest_and_smallest_compare_areas() -> None:
    masks, scores = candidate_masks()

    assert choose_mask(masks, scores, "largest") == 1
    assert choose_mask(masks, scores, "smallest") == 0


def test_overlay_changes_only_masked_pixels() -> None:
    image = Image.new("RGB", (10, 10), (10, 20, 30))
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True

    result = np.array(render_region_overlay(image, mask))

    assert (result[0, 0] == (10, 20, 30)).all()
    assert (result[9, 9] == (10, 20, 30)).all()
    assert (result[3, 3] == (255, 255, 255)).all()
    assert not (result[5, 5] == (10, 20, 30)).all()
    assert not (result[5, 5] == (255, 255, 255)).all()


@pytest.mark.slow
@pytest.mark.integration
def test_segmenter_reuses_embedding(
    installed_checkpoints: Callable[[Iterable[CheckpointSpec]], Path],
) -> None:
    model_dir = installed_checkpoints([MOBILE_SAM_CHECKPOINT])
    weights = checkpoint_path(model_dir, MOBILE_SAM_CHECKPOINT)
    sam = sam_model_registry["vit_t"](checkpoint=str(weights))
    segmenter = PromptedSegmenter(sam, torch.device("cpu"))
    set_image_calls = 0
    original_set_image = segmenter.predictor.set_image

    def counting_set_image(*args: object, **kwargs: object) -> None:
        nonlocal set_image_calls
        set_image_calls += 1
        original_set_image(*args, **kwargs)

    segmenter.predictor.set_image = counting_set_image  # type: ignore[method-assign]
    image = Image.new("RGB", (256, 192), (230, 230, 240))
    pixels = np.array(image)
    pixels[40:150, 80:200] = (150, 60, 50)
    image = Image.fromarray(pixels)

    mask, _ = segmenter.segment(image, RegionPrompt.from_lists([0.25, 0.15, 0.85, 0.85], None))
    segmenter.segment(image, RegionPrompt.from_lists(None, [0.5, 0.5]))

    assert set_image_calls == 1
    assert 0.10 < mask.mean() < 0.40
