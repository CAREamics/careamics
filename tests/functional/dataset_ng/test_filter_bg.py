from collections.abc import Sequence

import numpy as np
import pytest
from conftest import track_patching

from careamics.dataset_ng.factory import init_patch_extractor
from careamics.dataset_ng.filter_bg import (
    filter_background,
    filter_background_with_mask,
)
from careamics.dataset_ng.image_stack_loader import load_arrays
from careamics.dataset_ng.patch_extractor import PatchExtractor
from careamics.dataset_ng.patch_filter import MaskCoordFilter, MaxPatchFilter
from careamics.dataset_ng.patching_strategies import StratifiedPatchingStrategy


@pytest.mark.parametrize(
    "data_shapes,patch_size",
    [
        [[(2, 1, 64, 64), (1, 1, 43, 71), (3, 1, 35, 12)], (8, 8)],
        [[(2, 1, 64, 64, 64), (1, 1, 43, 71, 46), (3, 1, 35, 21, 12)], (8, 8, 8)],
    ],
)
def test_filter_background(
    data_shapes: Sequence[Sequence[int]], patch_size: Sequence[int]
):
    # data set-up
    rng = np.random.default_rng(42)
    axes = "SCYX" if len(patch_size) == 2 else "SCZYX"
    data = [rng.normal(1, 0.01, size=shape) for shape in data_shapes]
    # make the first sample much lower than the rest
    data_idx, sample_idx = 0, 0
    data[data_idx][sample_idx] *= 0.01

    # set up components
    patch_filter = MaxPatchFilter(threshold=0.8, threshold_ratio=0.75)
    patch_extractor = init_patch_extractor(PatchExtractor, load_arrays, data, axes)
    patching = StratifiedPatchingStrategy(
        patch_extractor.shapes, patch_size=patch_size, seed=42
    )
    background_prob = 0.1
    filter_background(
        patching,
        patch_extractor,
        patch_filter,
        ref_channel=0,
        bg_relative_prob=background_prob,
    )

    epochs = 100
    tracking_arrays = track_patching(patching, data_shapes, patch_size, epochs)
    # assert first sample has lower prob
    for data_idx, tracking_array in enumerate(tracking_arrays):
        for sample_idx, sample_tracking_array in enumerate(tracking_array):
            mean_pixel_prob = np.mean(sample_tracking_array) / epochs
            if data_idx == 0 and sample_idx == 0:
                # NOTE: probability depends on n_patches which is calculated as the ceil
                # means the resultant probability won't be exactly that set.
                np.testing.assert_allclose(mean_pixel_prob, background_prob, rtol=0.1)
            else:
                np.testing.assert_allclose(mean_pixel_prob, 1.0, rtol=0.1)


@pytest.mark.parametrize(
    "data_shapes,patch_size",
    [
        [[(2, 1, 64, 64), (1, 1, 43, 71), (3, 1, 35, 12)], (8, 8)],
        [[(2, 1, 64, 64, 64), (1, 1, 43, 71, 46), (3, 1, 35, 21, 12)], (8, 8, 8)],
    ],
)
def test_filter_background_w_mask(
    data_shapes: Sequence[Sequence[int]], patch_size: Sequence[int]
):
    # mask set-up
    axes = "SCYX" if len(patch_size) == 2 else "SCZYX"
    # mask the first sample
    data_idx, sample_idx = 0, 0
    masks = [np.ones(shape, dtype=bool) for shape in data_shapes]
    masks[data_idx][sample_idx][...] = False

    # set up components
    mask_extractor = init_patch_extractor(PatchExtractor, load_arrays, masks, axes)
    mask_filter = MaskCoordFilter(mask_extractor, coverage=0.25)
    patching = StratifiedPatchingStrategy(
        mask_extractor.shapes, patch_size=patch_size, seed=42
    )
    background_prob = 0.1
    filter_background_with_mask(
        patching,
        mask_filter,
        bg_relative_prob=background_prob,
    )

    epochs = 100
    tracking_arrays = track_patching(patching, data_shapes, patch_size, epochs)
    # assert first sample has lower prob
    for data_idx, tracking_array in enumerate(tracking_arrays):
        for sample_idx, sample_tracking_array in enumerate(tracking_array):
            mean_pixel_prob = np.mean(sample_tracking_array) / epochs
            if data_idx == 0 and sample_idx == 0:
                # NOTE: probability depends on n_patches which is calculated as the ceil
                # means the resultant probability won't be exactly that set.
                np.testing.assert_allclose(mean_pixel_prob, background_prob, rtol=0.1)
            else:
                np.testing.assert_allclose(mean_pixel_prob, 1.0, rtol=0.1)
