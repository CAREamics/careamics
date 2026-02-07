"""Tests for CAREamistV2."""

from pathlib import Path

import numpy as np
import pytest
import tifffile
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from careamics.careamist_v2 import CAREamistV2
from careamics.config.ng_factories.n2v_factory import create_advanced_n2v_config
from careamics.lightning.callbacks import CareamicsCheckpointInfo
from careamics.lightning.dataset_ng.data_module import CareamicsDataModule
from careamics.lightning.dataset_ng.lightning_modules import create_module


def random_array(shape: tuple[int, ...], seed: int = 42):
    """Return a random array with values between 0 and 255."""
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 255, shape)).astype(np.float32)


def test_v2_train_error_no_data(tmp_path: Path):
    """Test that a ValueError is raised when no training data is provided."""
    config = create_advanced_n2v_config(
        experiment_name="test",
        data_type="array",
        axes="YX",
        patch_size=(8, 8),
        batch_size=2,
        num_epochs=1,
        roi_size=5,
        masked_pixel_percentage=5,
    )
    careamist = CAREamistV2(config=config, work_dir=tmp_path)

    with pytest.raises(ValueError, match="Training data must be provided"):
        careamist.train()


def test_target_unsupported_warning_n2v(tmp_path: Path):
    """Test that a warning is emitted when a target is provided for N2V."""
    config = create_advanced_n2v_config(
        experiment_name="test",
        data_type="array",
        axes="YX",
        patch_size=(8, 8),
        batch_size=2,
        num_epochs=1,
        roi_size=5,
        masked_pixel_percentage=5,
    )
    careamics = CAREamistV2(config=config, work_dir=tmp_path)

    train_array = np.ones((32, 32))
    val_array = np.ones((32, 32))

    with pytest.warns(match="train_data_target.*ignored"):
        careamics.train(
            train_data=train_array,
            val_data=val_array,
            train_data_target=train_array,
        )


def test_v2_checkpoint_params_propagated(tmp_path: Path):
    """Test that checkpoint callback config is propagated to the ModelCheckpoint."""
    config = create_advanced_n2v_config(
        experiment_name="test",
        data_type="array",
        axes="YX",
        patch_size=(8, 8),
        batch_size=2,
        num_epochs=1,
        roi_size=5,
        masked_pixel_percentage=5,
    )
    config.training_config.checkpoint_callback.save_top_k = 2
    config.training_config.checkpoint_callback.monitor = "val_loss"
    config.training_config.checkpoint_callback.mode = "min"
    config.training_config.checkpoint_callback.save_last = True
    config.training_config.checkpoint_callback.every_n_epochs = 1

    careamist = CAREamistV2(config=config, work_dir=tmp_path)

    ckpt_callbacks = [
        cb for cb in careamist.callbacks if isinstance(cb, ModelCheckpoint)
    ]
    assert len(ckpt_callbacks) == 1
    ckpt_cb = ckpt_callbacks[0]

    assert ckpt_cb.save_top_k == 2
    assert ckpt_cb.monitor == "val_loss"
    assert ckpt_cb.mode == "min"
    assert ckpt_cb.save_last is True
    assert ckpt_cb.every_n_epochs == 1
    assert ckpt_cb.dirpath == str(tmp_path / "checkpoints")


@pytest.mark.mps_gh_fail
def test_v2_train_array(tmp_path: Path):
    """Test that CAREamistV2 can be trained on arrays."""
    train_array = random_array((32, 32), seed=42)
    val_array = random_array((32, 32), seed=43)

    config = create_advanced_n2v_config(
        experiment_name="test",
        data_type="array",
        axes="YX",
        patch_size=(8, 8),
        batch_size=2,
        num_epochs=1,
        roi_size=5,
        masked_pixel_percentage=5,
    )

    careamist = CAREamistV2(config=config, work_dir=tmp_path)
    careamist.train(train_data=train_array, val_data=val_array)

    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()


@pytest.mark.mps_gh_fail
@pytest.mark.parametrize("independent_channels", [False, True])
def test_v2_train_array_channel(tmp_path: Path, independent_channels: bool):
    """Test that CAREamistV2 can be trained on arrays with channels."""
    train_array = random_array((32, 32, 3), seed=42)
    val_array = random_array((32, 32, 3), seed=43)

    config = create_advanced_n2v_config(
        experiment_name="test",
        data_type="array",
        axes="YXC",
        patch_size=(8, 8),
        batch_size=2,
        num_epochs=1,
        n_channels=3,
        roi_size=5,
        masked_pixel_percentage=5,
    )
    config.algorithm_config.model.in_channels = 3
    config.algorithm_config.model.num_classes = 3
    config.algorithm_config.model.independent_channels = independent_channels

    careamist = CAREamistV2(config=config, work_dir=tmp_path)
    careamist.train(train_data=train_array, val_data=val_array)

    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()


@pytest.mark.mps_gh_fail
def test_v2_train_array_3d(tmp_path: Path):
    """Test that CAREamistV2 can be trained on 3D arrays."""
    train_array = random_array((8, 32, 32), seed=42)
    val_array = random_array((8, 32, 32), seed=43)

    config = create_advanced_n2v_config(
        experiment_name="test",
        data_type="array",
        axes="ZYX",
        patch_size=(8, 16, 16),
        batch_size=2,
        num_epochs=1,
        roi_size=5,
        masked_pixel_percentage=5,
    )

    careamist = CAREamistV2(config=config, work_dir=tmp_path)
    careamist.train(train_data=train_array, val_data=val_array)

    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()


@pytest.mark.mps_gh_fail
def test_v2_train_tiff_in_memory(tmp_path: Path):
    """Test that CAREamistV2 can be trained with tiff files in memory."""
    train_array = random_array((32, 32), seed=42)
    val_array = random_array((32, 32), seed=43)

    train_file = tmp_path / "train.tiff"
    tifffile.imwrite(train_file, train_array)

    val_file = tmp_path / "val.tiff"
    tifffile.imwrite(val_file, val_array)

    config = create_advanced_n2v_config(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=(8, 8),
        batch_size=2,
        num_epochs=1,
        roi_size=5,
        masked_pixel_percentage=5,
    )

    careamist = CAREamistV2(config=config, work_dir=tmp_path)
    careamist.train(train_data=train_file, val_data=val_file)

    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()


@pytest.mark.mps_gh_fail
def test_v2_train_tiff_not_in_memory(tmp_path: Path):
    """Test N2V training from tiff files with in_memory=False (on-disk reading)."""
    train_array = random_array((32, 32), seed=42)
    val_array = random_array((32, 32), seed=43)

    train_file = tmp_path / "train.tiff"
    tifffile.imwrite(train_file, train_array)

    val_file = tmp_path / "val.tiff"
    tifffile.imwrite(val_file, val_array)

    config = create_advanced_n2v_config(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=(8, 8),
        batch_size=2,
        num_epochs=1,
        roi_size=5,
        masked_pixel_percentage=5,
        in_memory=False,
    )

    careamist = CAREamistV2(config=config, work_dir=tmp_path)
    careamist.train(train_data=train_file, val_data=val_file)

    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()


@pytest.mark.mps_gh_fail
def test_load_checkpoint_with_careamics_checkpoint_info(tmp_path: Path) -> None:
    """Test loading a checkpoint saved with CareamicsCheckpointInfo callback."""
    # Create minimal configuration
    config = create_advanced_n2v_config(
        experiment_name="test_checkpoint_loading",
        data_type="array",
        axes="YX",
        patch_size=(16, 16),
        batch_size=2,
        num_epochs=1,
        roi_size=5,
        masked_pixel_percentage=5,
    )

    # Create module and datamodule
    module = create_module(config.algorithm_config)
    train_data = np.random.rand(32, 32).astype(np.float32)
    val_data = np.random.rand(16, 16).astype(np.float32)

    datamodule = CareamicsDataModule(
        data_config=config.data_config,
        train_data=train_data,
        val_data=val_data,
    )

    # Set up trainer with CareamicsCheckpointInfo callback
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="test_checkpoint_{epoch:02d}_{step}",
        save_last=True,
    )

    careamics_callback = CareamicsCheckpointInfo(
        careamics_version=config.version,
        experiment_name=config.experiment_name,
        training_config=config.training_config,
    )

    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        callbacks=[checkpoint_callback, careamics_callback],
        enable_progress_bar=False,
        logger=False,
    )

    # Train for one step to create a checkpoint
    trainer.fit(module, datamodule=datamodule)

    # Find the saved checkpoint
    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
    assert len(checkpoint_files) > 0, "No checkpoint was saved"
    # Prefer "last.ckpt" if it exists, otherwise use the first one
    checkpoint_path = (
        checkpoint_dir / "last.ckpt"
        if (checkpoint_dir / "last.ckpt").exists()
        else checkpoint_files[0]
    )

    # Load checkpoint
    loaded_config, loaded_module = CAREamistV2._from_checkpoint(checkpoint_path)

    # Verify configuration matches
    assert loaded_config.experiment_name == config.experiment_name
    assert loaded_config.version == config.version
    assert loaded_config.algorithm_config.algorithm == config.algorithm_config.algorithm
    assert loaded_config.data_config.axes == config.data_config.axes
    assert loaded_config.data_config.batch_size == config.data_config.batch_size
    assert list(loaded_config.data_config.patching.patch_size) == list(
        config.data_config.patching.patch_size
    )

    # Verify module was loaded
    assert loaded_module is not None
    assert type(loaded_module).__name__ == type(module).__name__
