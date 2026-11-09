from collections.abc import Sequence

import numpy as np
import pytest
from tests.functional.dataset.utils import track_patching

from careamics.config.data.data_config import DataConfig
from careamics.dataset.factory import (
    TrainValSplitData,
    create_tilt_axis_split_datasets,
)
from careamics.dataset.factory.val_split import create_tilt_axis_split, create_val_split
from careamics.dataset.patching import (
    StratifiedPatching,
)


# TODO move to unit
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
    Test that created train and validation patching strategies are complements.

    i.e. patches from each do not ever overlap.
    """
    rng = np.random.default_rng(42)
    patching_strategy = StratifiedPatching(data_shapes, patch_size, 42)

    n_val_patches = int(np.ceil(patching_strategy.n_patches * 0.1))  # 10% of patches
    train_strat, val_strat = create_val_split(patching_strategy, n_val_patches, rng)

    train_tracking_arrays = track_patching(
        train_strat, data_shapes, patch_size, epochs=5
    )
    val_tracking_arrays = track_patching(val_strat, data_shapes, patch_size, epochs=5)

    # check not all zeros
    assert any((train_array != 0).any() for train_array in train_tracking_arrays)
    assert any((val_array != 0).any() for val_array in val_tracking_arrays)
    for train_array, val_array in zip(
        train_tracking_arrays, val_tracking_arrays, strict=True
    ):
        # there should never be any pixels ever selected in both the train and val
        assert not np.logical_and(train_array != 0, val_array != 0).all()


@pytest.mark.parametrize("tilt_axis,axis_idx", [("Z", 0), ("Y", 1), ("X", 2)])
def test_tilt_axis_split_does_not_cross_boundary(tilt_axis: str, axis_idx: int):
    """Test that tilt-axis train and validation patches stay on their own side."""
    data_shapes = [(1, 1, 64, 64, 64)]
    patch_size = (8, 8, 8)
    patching_strategy = StratifiedPatching(data_shapes, patch_size, seed=42)
    train_patching, val_patching = create_tilt_axis_split(
        patching_strategy,
        tilt_axis=tilt_axis,
        validation_fraction=0.25,
    )
    cutoff = int(data_shapes[0][2 + axis_idx] * 0.75)

    assert train_patching.n_patches > 0
    assert val_patching.n_patches > 0

    for index in range(train_patching.n_patches):
        spec = train_patching.get_patch_spec(index)
        assert spec["coords"][axis_idx] + patch_size[axis_idx] <= cutoff

    for index in range(val_patching.n_patches):
        spec = val_patching.get_patch_spec(index)
        assert spec["coords"][axis_idx] >= cutoff
        assert spec["coords"][axis_idx] + patch_size[axis_idx] <= 64


def test_tilt_axis_split_allows_one_patch_per_side():
    """Allow a split where each side contains one full patch along the axis."""
    patching_strategy = StratifiedPatching([(1, 1, 16, 16, 16)], (8, 8, 8), seed=42)

    train_patching, val_patching = create_tilt_axis_split(
        patching_strategy,
        tilt_axis="Z",
        validation_fraction=0.5,
    )

    assert train_patching.n_patches > 0
    assert val_patching.n_patches > 0


def test_create_tilt_axis_split_datasets():
    """Test that train and validation datasets using a tilt-axis split."""
    config = DataConfig(
        mode="training",
        data_type="array",
        axes="SCZYX",
        patching={"name": "stratified", "patch_size": (8, 8, 8), "seed": 42},
        normalization={"name": "none"},
        batch_size=1,
    )
    data = np.zeros((1, 1, 64, 64, 64), dtype=np.float32)

    train_dataset, val_dataset = create_tilt_axis_split_datasets(
        config=config,
        data=TrainValSplitData(train_data=[data], n_val_patches=0),
        loading=None,
        tilt_axis="Y",
        validation_fraction=0.25,
    )

    assert len(train_dataset) > 0
    assert len(val_dataset) > 0
    train_spec = train_dataset.patch_constructor.patching_strategy.get_patch_spec(0)
    val_spec = val_dataset.patch_constructor.patching_strategy.get_patch_spec(0)
    assert train_spec["coords"][1] + 8 <= 48
    assert val_spec["coords"][1] >= 48
