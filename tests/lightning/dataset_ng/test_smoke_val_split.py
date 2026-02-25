from collections.abc import Sequence

import numpy as np
import pytest
from pytorch_lightning import Callback, Trainer

from careamics.config.data.patching_strategies import StratifiedPatchingConfig
from careamics.config.ng_factories import create_advanced_n2v_config
from careamics.lightning.dataset_ng.data_module import CareamicsDataModule
from careamics.lightning.dataset_ng.lightning_modules import N2VModule


class _PatchTrackerCallback(Callback):
    """Tracks the location of selected patches during validation and training."""

    def __init__(self, data_shapes: Sequence[Sequence[int]]):
        super().__init__()
        self.train_tracking_arrays = [
            np.zeros(shape, dtype=bool) for shape in data_shapes
        ]
        self.val_tracking_arrays = [
            np.zeros(shape, dtype=bool) for shape in data_shapes
        ]

    def on_train_batch_start(
        self,
        trainer,
        pl_module,
        batch,  #: ImageRegionData[PatchSpecs],
        batch_idx: int,
    ) -> None:
        for data_idx, patch_slice in self.batch_patch_slices(batch):
            self.train_tracking_arrays[data_idx][patch_slice] = True

    def on_validation_batch_start(
        self,
        trainer,
        pl_module,
        batch,  #: ImageRegionData[PatchSpecs],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        for data_idx, patch_slice in self.batch_patch_slices(batch):
            self.val_tracking_arrays[data_idx][patch_slice] = True

    @staticmethod
    def batch_patch_slices(batch):
        batch = batch[0]  # why is this a list of one
        n = batch.data.shape[0]
        region_specs = batch.region_spec
        for i in range(n):
            data_idx = region_specs["data_idx"][i]
            sample_idx = region_specs["sample_idx"][i]
            coords = tuple(c for c in region_specs["coords"][i])
            patch_shape = tuple(ps for ps in region_specs["patch_size"][i])

            patch_slice = (
                sample_idx,
                ...,
                *(slice(c, c + ps) for c, ps in zip(coords, patch_shape, strict=True)),
            )
            yield data_idx, patch_slice


@pytest.mark.parametrize(
    "data_shapes,patch_size",
    [
        [[(2, 1, 64, 64), (1, 1, 43, 71), (3, 1, 14, 17)], (8, 8)],
        [[(2, 1, 64, 64, 64), (1, 1, 43, 71, 46), (3, 1, 9, 17, 12)], (8, 8, 8)],
    ],
    ids=["2D", "3D"],
)
def test_smoke_val_split(
    tmp_path, data_shapes: Sequence[Sequence[int]], patch_size: Sequence[int]
):
    """
    Test that the validation and train patches from the data module do not overlap.
    """

    rng = np.random.default_rng(42)

    # training data
    train_arrays = [
        rng.integers(0, 255, shape).astype(np.float32) for shape in data_shapes
    ]

    if len(patch_size) == 2:
        axes = "SCYX"
    else:
        axes = "SCZYX"

    config = create_advanced_n2v_config(
        "smoke_val_split",
        data_type="array",
        axes=axes,
        patch_size=patch_size,
        batch_size=2,
        num_epochs=5,
        n_channels=1,
        augmentations=[],
        # prevent validation error
        masked_pixel_percentage=3,
    )
    config.data_config.patching = StratifiedPatchingConfig(
        patch_size=patch_size, seed=42
    )

    model = N2VModule(algorithm_config=config.algorithm_config)
    data = CareamicsDataModule(
        data_config=config.data_config,
        train_data=train_arrays,
        # val splitting
        val_percentage=0.1,
        val_minimum_split=5,
    )
    tracking_callback = _PatchTrackerCallback(data_shapes)

    # create trainer
    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmp_path,
        callbacks=[tracking_callback],
    )

    # train
    trainer.fit(model, datamodule=data)

    # now assert that the tiles selected for train and validation do not overlap
    train_tracking_arrays = tracking_callback.train_tracking_arrays
    val_tracking_arrays = tracking_callback.val_tracking_arrays
    # check not all zeros
    assert any((train_array != 0).any() for train_array in train_tracking_arrays)
    assert any((val_array != 0).any() for val_array in val_tracking_arrays)
    for train_array, val_array in zip(
        train_tracking_arrays, val_tracking_arrays, strict=True
    ):
        # there should never be any pixels ever selected in both the train and val
        assert not np.logical_and(train_array != 0, val_array != 0).all()
