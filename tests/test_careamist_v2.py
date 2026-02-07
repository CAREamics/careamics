from pathlib import Path

import numpy as np
import pytest
import tifffile

from careamics.careamist_v2 import CAREamistV2
from careamics.config.ng_factories.n2v_factory import create_advanced_n2v_config


def random_array(shape: tuple[int, ...], seed: int = 42):
    """Return a random array with values between 0 and 255."""
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 255, shape)).astype(np.float32)


def test_v2_train_error_target_unsupervised_algorithm(
    tmp_path: Path, minimum_n2v_configuration: dict
):
    """Test that an error is raised when a target is provided for N2V."""
    # create configuration using factory
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

    # train error with Paths
    careamics = CAREamistV2(config=config, work_dir=tmp_path)
    with pytest.raises(ValueError):
        careamics.train(
            train_data=tmp_path,
            train_data_target=tmp_path,
        )

    # train error with strings
    with pytest.raises(ValueError):
        careamics.train(
            train_data=str(tmp_path),
            train_data_target=str(tmp_path),
        )

    # train error with arrays
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
    with pytest.raises(ValueError):
        careamics.train(
            train_data=np.ones((32, 32)),
            train_data_target=np.ones((32, 32)),
        )


@pytest.mark.skip("val splits not implemented")
@pytest.mark.mps_gh_fail
def test_v2_train_single_array_no_val(tmp_path: Path, minimum_n2v_configuration: dict):
    """Test that CAREamistV2 can be trained with arrays."""
    # training data
    train_array = random_array((32, 32))

    # create configuration
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

    # instantiate CAREamistV2
    careamist = CAREamistV2(config=config, work_dir=tmp_path)

    # train CAREamistV2
    careamist.train(train_data=train_array)

    # check that it trained
    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()


@pytest.mark.skip("val splits not implemented")
@pytest.mark.mps_gh_fail
def test_v2_train_array(tmp_path: Path):
    """Test that CAREamistV2 can be trained on arrays."""
    # training data
    train_array = random_array((32, 32))
    val_array = random_array((32, 32))

    # create configuration
    config = create_advanced_n2v_config(
        experiment_name="test",
        data_type="array",
        axes="YX",
        patch_size=(8, 8),
        batch_size=2,
        num_epochs=6,
        roi_size=5,
        masked_pixel_percentage=5,
    )
    config.training_config.checkpoint_callback.save_top_k = 2
    config.training_config.checkpoint_callback.monitor = "val_loss"
    config.training_config.checkpoint_callback.mode = "min"
    config.training_config.checkpoint_callback.save_last = True
    config.training_config.checkpoint_callback.every_n_epochs = 1

    # instantiate CAREamistV2
    careamist = CAREamistV2(config=config, work_dir=tmp_path)

    # train CAREamistV2
    careamist.train(train_data=train_array, val_data=val_array)

    # check that it trained
    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()

    # check save_top_k=2 functionality
    checkpoint_files = list((tmp_path / "checkpoints").glob("*.ckpt"))
    assert (
        len(checkpoint_files) == 3
    ), f"Expected 3 checkpoint files (2 best + last), found {len(checkpoint_files)}"

    # Verify last.ckpt exists
    last_ckpt_exists = any(f.name == "last.ckpt" for f in checkpoint_files)
    assert last_ckpt_exists, "last.ckpt should exist when save_last=True"

    # Verify we have exactly 2 non-last checkpoint files
    non_last_checkpoints = [f for f in checkpoint_files if f.name != "last.ckpt"]
    assert (
        len(non_last_checkpoints) == 2
    ), f"Expected exactly 2 best checkpoints, found {len(non_last_checkpoints)}"


@pytest.mark.skip("val splits not implemented")
@pytest.mark.mps_gh_fail
@pytest.mark.parametrize("independent_channels", [False, True])
def test_v2_train_array_channel(
    tmp_path: Path, minimum_n2v_configuration: dict, independent_channels: bool
):
    """Test that CAREamistV2 can be trained on arrays with channels."""
    # training data
    train_array = random_array((32, 32, 3))
    val_array = random_array((32, 32, 3))

    # create configuration
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

    # instantiate CAREamistV2
    careamist = CAREamistV2(config=config, work_dir=tmp_path)

    # train CAREamistV2
    careamist.train(train_data=train_array, val_data=val_array)

    # check that it trained
    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()


@pytest.mark.skip("val splits not implemented")
@pytest.mark.mps_gh_fail
def test_v2_train_array_3d(tmp_path: Path, minimum_n2v_configuration: dict):
    """Test that CAREamistV2 can be trained on 3D arrays."""
    # training data
    train_array = random_array((8, 32, 32))
    val_array = random_array((8, 32, 32))

    # create configuration
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

    # instantiate CAREamistV2
    careamist = CAREamistV2(config=config, work_dir=tmp_path)

    # train CAREamistV2
    careamist.train(train_data=train_array, val_data=val_array)

    # check that it trained
    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()


@pytest.mark.skip("val splits not implemented")
@pytest.mark.mps_gh_fail
def test_v2_train_tiff_files_in_memory_no_val(
    tmp_path: Path, minimum_n2v_configuration: dict
):
    """Test that CAREamistV2 can be trained with tiff files in memory."""
    # training data
    train_array = random_array((32, 32))

    # save files
    train_file = tmp_path / "train.tiff"
    tifffile.imwrite(train_file, train_array)

    # create configuration
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

    # instantiate CAREamistV2
    careamist = CAREamistV2(config=config, work_dir=tmp_path)

    # train CAREamistV2
    careamist.train(train_data=train_file)

    # check that it trained
    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()


@pytest.mark.skip("val splits not implemented")
@pytest.mark.mps_gh_fail
def test_v2_train_tiff_files_in_memory(tmp_path: Path, minimum_n2v_configuration: dict):
    """Test that CAREamistV2 can be trained with tiff files in memory."""
    # training data
    train_array = random_array((32, 32))
    val_array = random_array((32, 32))

    # save files
    train_file = tmp_path / "train.tiff"
    tifffile.imwrite(train_file, train_array)

    val_file = tmp_path / "val.tiff"
    tifffile.imwrite(val_file, val_array)

    # create configuration
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

    # instantiate CAREamistV2
    careamist = CAREamistV2(config=config, work_dir=tmp_path)

    # train CAREamistV2
    careamist.train(train_data=train_file, val_data=val_file)

    # check that it trained
    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()


@pytest.mark.skip("val splits not implemented")
@pytest.mark.mps_gh_fail
def test_v2_train_tiff_files(tmp_path: Path, minimum_n2v_configuration: dict):
    """Test that CAREamistV2 can be trained with tiff files by deactivating
    the in memory dataset.
    """
    # training data
    train_array = random_array((32, 32))
    val_array = random_array((32, 32))

    # save files
    train_file = tmp_path / "train.tiff"
    tifffile.imwrite(train_file, train_array)

    val_file = tmp_path / "val.tiff"
    tifffile.imwrite(val_file, val_array)

    # create configuration
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

    # instantiate CAREamistV2
    careamist = CAREamistV2(config=config, work_dir=tmp_path)

    # train CAREamistV2
    careamist.train(train_data=train_file, val_data=val_file, use_in_memory=False)

    # check that it trained
    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()
