"""Tests for CAREamistV2."""

from pathlib import Path

import numpy as np
import pytest
import tifffile

from careamics.careamist_v2 import CAREamistV2
from careamics.config.ng_factories.care_n2n_factory import create_advanced_care_config
from careamics.config.ng_factories.n2v_factory import create_advanced_n2v_config


def random_array(shape: tuple[int, ...], seed: int = 42):
    """Return a random array with values between 0 and 255."""
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 255, shape)).astype(np.float32)


def test_train_error_no_data(tmp_path: Path):
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


def test_train_error_no_target_data(tmp_path: Path):
    """Test that a ValueError is raised when no training data is provided."""
    train_array = np.ones((32, 32))
    train_target_array = np.ones((32, 32))
    val_array = np.ones((32, 32))

    config = create_advanced_care_config(
        experiment_name="test",
        data_type="array",
        axes="YX",
        patch_size=(8, 8),
        batch_size=2,
        num_epochs=1,
    )
    careamist = CAREamistV2(config=config, work_dir=tmp_path)

    with pytest.raises(ValueError, match="Training target data must be provided"):
        careamist.train(train_data=train_array, val_data=val_array)
    with pytest.raises(ValueError, match="Validation target data must be provided"):
        careamist.train(
            train_data=train_array,
            train_data_target=train_target_array,
            val_data=val_array,
        )


# TODO since this happens in the LightningModule, this test should be removed
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
            val_data_target=val_array,
        )


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

    assert Path(tmp_path / "checkpoints" / "test" / "test_last.ckpt").exists()


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

    assert Path(tmp_path / "checkpoints" / "test" / "test_last.ckpt").exists()


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

    assert Path(tmp_path / "checkpoints" / "test" / "test_last.ckpt").exists()


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

    assert Path(tmp_path / "checkpoints" / "test" / "test_last.ckpt").exists()


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

    assert Path(tmp_path / "checkpoints" / "test" / "test_last.ckpt").exists()


def test_init_from_checkpoint(tmp_path: Path, checkpoint):
    """Ensure CAREamist can be initialised from a checkpoint."""
    checkpoint_path, expected_module_type, expected_config = checkpoint

    careamist = CAREamistV2(checkpoint_path=checkpoint_path, work_dir=tmp_path)

    assert isinstance(careamist.model, expected_module_type)

    # careamist by default enables progress bar during initialization
    expected_config.training_config.lightning_trainer_config["enable_progress_bar"] = (
        True
    )
    assert careamist.config == expected_config
