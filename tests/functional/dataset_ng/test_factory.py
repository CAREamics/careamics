from collections.abc import Sequence

import numpy as np
import pytest
from tests.functional.dataset_ng.utils import assert_expected_pixel_probability
from tests.utils import (
    coord_filter_dict_testing,
    ng_data_config_dict_testing,
    patch_filter_dict_testing,
)

from careamics.config.data.ng_data_config import NGDataConfig
from careamics.dataset_ng.factory import TrainValSplitData, create_train_dataset


@pytest.mark.parametrize(
    "data_shapes,patch_size",
    [
        [[(2, 1, 64, 64), (1, 1, 43, 71), (3, 1, 35, 12)], (8, 8)],
        [[(2, 1, 64, 64, 64), (1, 1, 43, 71, 46), (3, 1, 35, 21, 12)], (8, 8, 8)],
    ],
)
@pytest.mark.parametrize("background_prob", [0, 0.1, 0.8])
def test_filter_background(
    data_shapes: Sequence[Sequence[int]],
    patch_size: tuple[int, ...],
    background_prob: float,
):
    threshold = 0.5
    axes = "SCYX" if len(patch_size) == 2 else "SCZYX"
    config_dict = ng_data_config_dict_testing(
        mode="training",
        axes=axes,
        patch_size=patch_size,
        filtered_patch_prob=background_prob,
    )
    patch_filter = patch_filter_dict_testing(name="max")
    patch_filter["threshold"] = threshold
    patch_filter["threshold_ratio"] = 0.75
    config_dict["patch_filter"] = patch_filter
    config = NGDataConfig(**config_dict)

    # data set-up
    rng = np.random.default_rng(42)
    data = [rng.normal(1, 0.01, size=shape) for shape in data_shapes]
    # make the first sample much lower than the rest
    data_idx, sample_idx = 0, 0
    data[data_idx][sample_idx] *= threshold - threshold / 2
    data_input = TrainValSplitData(data, n_val_patches=0)
    mean_expected_prob = {(0, 0): background_prob}

    dataset = create_train_dataset(config, data_input, loading=None)
    patching = dataset.patching_strategy

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
@pytest.mark.parametrize("background_prob", [0, 0.1, 0.8])
def test_filter_background_w_mask(
    data_shapes: Sequence[Sequence[int]],
    patch_size: tuple[int, ...],
    background_prob: float,
):
    axes = "SCYX" if len(patch_size) == 2 else "SCZYX"
    config_dict = ng_data_config_dict_testing(
        mode="training",
        axes=axes,
        patch_size=patch_size,
        filtered_patch_prob=background_prob,
    )
    coord_filter = coord_filter_dict_testing()
    config_dict["coord_filter"] = coord_filter
    config = NGDataConfig(**config_dict)

    # data set-up
    rng = np.random.default_rng(42)
    data = [rng.normal(1, 0.01, size=shape) for shape in data_shapes]
    data_idx, sample_idx = 0, 0

    # mask the first sample
    masks = [np.ones(shape, dtype=bool) for shape in data_shapes]
    masks[data_idx][sample_idx][...] = False
    data_input = TrainValSplitData(data, n_val_patches=0, train_data_mask=masks)
    mean_expected_prob = {(0, 0): background_prob}

    dataset = create_train_dataset(config, data_input, loading=None)
    patching = dataset.patching_strategy

    assert_expected_pixel_probability(
        patching, data_shapes, patch_size, mean_expected_prob
    )
