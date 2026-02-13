from collections.abc import Sequence

import numpy as np
import pytest

from careamics.dataset_ng.patching_strategies import (
    StratifiedPatchingStrategy,
)
from careamics.dataset_ng.val_split import create_val_split


@pytest.mark.parametrize(
    "data_shapes,patch_size",
    [
        [[(2, 1, 64, 64), (1, 1, 43, 71), (3, 1, 16, 9)], (8, 8)],
        [[(2, 1, 64, 64), (1, 1, 43, 71), (3, 1, 14, 9)], (8, 5)],
        [[(2, 1, 64, 64, 64), (1, 1, 43, 71, 46), (3, 1, 16, 9, 12)], (8, 8, 8)],
        [[(2, 1, 64, 64, 64), (1, 1, 43, 71, 46), (3, 1, 14, 9, 12)], (8, 5, 7)],
    ],
)
def test_train_val_complementary(
    data_shapes: Sequence[Sequence[int]], patch_size: Sequence[int]
):
    """
    Ensure created train and validation patching strategies are complements.

    i.e. patches from each do not ever overlap.
    """
    rng = np.random.default_rng(42)
    patching_strategy = StratifiedPatchingStrategy(data_shapes, patch_size, 42)

    n_val_patches = int(np.ceil(patching_strategy.n_patches * 0.1))  # 10% of patches
    train_strat, val_strat = create_val_split(patching_strategy, n_val_patches, rng)

    train_tracking_arrays = [np.zeros(shape, dtype=int) for shape in data_shapes]
    val_tracking_arrays = [np.zeros(shape, dtype=int) for shape in data_shapes]

    epochs = 5
    for _ in range(epochs):
        for i in range(train_strat.n_patches):
            patch_spec = train_strat.get_patch_spec(i)
            data_idx = patch_spec["data_idx"]
            sample_idx = patch_spec["sample_idx"]
            coord = patch_spec["coords"]
            patch_slice = [
                slice(c, c + ps) for c, ps in zip(coord, patch_size, strict=True)
            ]
            train_tracking_arrays[data_idx][sample_idx, ..., *patch_slice] += 1
        for i in range(val_strat.n_patches):
            patch_spec = val_strat.get_patch_spec(i)
            data_idx = patch_spec["data_idx"]
            sample_idx = patch_spec["sample_idx"]
            coord = patch_spec["coords"]
            patch_slice = [
                slice(c, c + ps) for c, ps in zip(coord, patch_size, strict=True)
            ]
            val_tracking_arrays[data_idx][sample_idx, ..., *patch_slice] += 1

    # check not all zeros
    assert any((train_array != 0).any() for train_array in train_tracking_arrays)
    assert any((val_array != 0).any() for val_array in val_tracking_arrays)
    for train_array, val_array in zip(
        train_tracking_arrays, val_tracking_arrays, strict=True
    ):
        # there should never be any pixels ever selected in both the train and val
        assert not np.logical_and(train_array != 0, val_array != 0).all()
