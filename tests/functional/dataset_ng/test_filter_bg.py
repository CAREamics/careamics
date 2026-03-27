from collections.abc import Sequence

import numpy as np
import pytest
from tests.functional.dataset_ng.utils import (
    assert_expected_pixel_probability,
)

from careamics.config.data.patch_filter import MaskFilterConfig, MaxFilterConfig
from careamics.dataset_ng.factory import init_patch_extractor
from careamics.dataset_ng.filter_bg import (
    filter_background,
    filter_background_with_mask,
)
from careamics.dataset_ng.image_stack_loader import load_arrays
from careamics.dataset_ng.patch_extractor import PatchExtractor
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
    ndims = len(patch_size)
    axes = "SCYX" if ndims == 2 else "SCZYX"
    data = [rng.normal(1, 0.01, size=shape) for shape in data_shapes]
    # make the first sample much lower than the rest
    data_idx, sample_idx = 0, 0
    data[data_idx][sample_idx] *= 0.01

    # set up components
    patch_filter_config = MaxFilterConfig(threshold=0.8, coverage=(1 / (2**ndims)))
    patch_extractor = init_patch_extractor(PatchExtractor, load_arrays, data, axes)
    patching = StratifiedPatchingStrategy(
        patch_extractor.shapes, patch_size=patch_size, seed=42
    )
    background_prob = 0.1
    filter_background(
        patching,
        patch_extractor,
        patch_filter_config,
        ref_channel=0,
        bg_relative_prob=background_prob,
    )

    mean_expected_prob = {(0, 0): background_prob}
    assert_expected_pixel_probability(
        patching, data_shapes, patch_size, mean_expected_prob
    )


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
    ndims = len(patch_size)
    axes = "SCYX" if ndims == 2 else "SCZYX"
    # mask the first sample
    data_idx, sample_idx = 0, 0
    masks = [np.ones(shape, dtype=bool) for shape in data_shapes]
    masks[data_idx][sample_idx][...] = False

    # set up components
    mask_extractor = init_patch_extractor(PatchExtractor, load_arrays, masks, axes)
    mask_filter_config = MaskFilterConfig(coverage=(1 / 2**ndims))
    patching = StratifiedPatchingStrategy(
        mask_extractor.shapes, patch_size=patch_size, seed=42
    )
    background_prob = 0.1
    filter_background_with_mask(
        patching,
        mask_filter_config,
        mask_extractor,
        bg_relative_prob=background_prob,
    )

    mean_expected_prob = {(0, 0): background_prob}
    assert_expected_pixel_probability(
        patching, data_shapes, patch_size, mean_expected_prob
    )
