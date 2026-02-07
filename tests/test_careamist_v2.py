"""Tests for CAREamistV2."""

from pathlib import Path

import numpy as np
import pytest
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from careamics.careamist_v2 import CAREamistV2
from careamics.config.ng_factories import create_advanced_n2v_config
from careamics.lightning.callbacks import CareamicsCheckpointInfo
from careamics.lightning.dataset_ng.data_module import CareamicsDataModule
from careamics.lightning.dataset_ng.lightning_modules import create_module


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
