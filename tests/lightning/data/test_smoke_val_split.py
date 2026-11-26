import logging
from collections.abc import Sequence

import numpy as np
import pytest
import torch
from lightning.pytorch import Callback, Trainer

from careamics.config.data.patching_strategies import StratifiedPatchingConfig
from careamics.config.factories import create_advanced_n2v_config
from careamics.lightning.data import CareamicsDataModule
from careamics.lightning.modules import N2VModule


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
        n_val_patches=8,
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


def _train_array() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (2, 1, 64, 64)).astype(np.float32)


def _n2v_datamodule(train_data, val_data, seed: int):
    config = create_advanced_n2v_config(
        "test",
        data_type="array",
        axes="SCYX",
        patch_size=(8, 8),
        batch_size=2,
        num_epochs=1,
        n_channels=1,
        n_val_patches=4,
        masked_pixel_percentage=3,
        seed=seed,
    )
    datamodule = CareamicsDataModule(
        data_config=config.data_config, train_data=train_data, val_data=val_data
    )
    return config, datamodule


def _fit_n2v(tmp_path, val_data, seed: int = 42):
    train_data = _train_array()
    config, datamodule = _n2v_datamodule(train_data, val_data, seed)
    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmp_path,
        logger=False,
        enable_progress_bar=False,
    )
    trainer.fit(N2VModule(config.algorithm_config), datamodule=datamodule)
    ckpt_path = tmp_path / "last.ckpt"
    trainer.save_checkpoint(ckpt_path)
    checkpoint = torch.load(ckpt_path, weights_only=False)
    return datamodule, checkpoint, train_data, ckpt_path


def _resume(tmp_path, train_data, seed: int, ckpt_path):
    config, datamodule = _n2v_datamodule(train_data, None, seed)
    trainer = Trainer(
        max_epochs=2,
        default_root_dir=tmp_path,
        logger=False,
        enable_progress_bar=False,
    )
    trainer.fit(
        N2VModule(config.algorithm_config),
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )
    return datamodule


def test_val_patch_specs_recorded(tmp_path):
    """Test that automatically split validation patches are saved in the checkpoint."""
    datamodule, checkpoint, _, _ = _fit_n2v(tmp_path, val_data=None)
    state = checkpoint[CareamicsDataModule.__qualname__]

    assert datamodule.val_patch_specs is not None
    assert len(datamodule.val_patch_specs) == 4
    assert set(datamodule.val_patch_specs[0]) == {
        "data_idx",
        "sample_idx",
        "coords",
        "patch_size",
        "source",
    }
    assert state["val_patch_specs"] == datamodule.val_patch_specs


def test_val_patch_specs_not_recorded_with_val_data(tmp_path):
    """Test that no validation patches are recorded when validation data is given."""
    val_data = np.zeros((1, 1, 64, 64), dtype=np.float32)
    datamodule, checkpoint, _, _ = _fit_n2v(tmp_path, val_data=val_data)

    assert datamodule.val_patch_specs is None
    assert CareamicsDataModule.__qualname__ not in checkpoint


def test_val_patch_specs_resume_warns_on_seed_mismatch(tmp_path, caplog):
    """Test that a different resume seed logs a warning and the same seed does not."""
    datamodule, _, train_data, ckpt_path = _fit_n2v(tmp_path, val_data=None)
    original = datamodule.val_patch_specs
    module_logger = logging.getLogger("careamics.lightning.data.data_module")
    module_logger.addHandler(caplog.handler)
    try:
        caplog.clear()
        resumed = _resume(
            tmp_path / "mismatch", train_data, seed=99, ckpt_path=ckpt_path
        )
        assert "Validation patches differ" in caplog.text
        assert resumed.val_patch_specs != original

        caplog.clear()
        resumed_same = _resume(
            tmp_path / "same", train_data, seed=42, ckpt_path=ckpt_path
        )
        assert "Validation patches differ" not in caplog.text
        assert resumed_same.val_patch_specs == original
    finally:
        module_logger.removeHandler(caplog.handler)
