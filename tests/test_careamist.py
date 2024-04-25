from pathlib import Path

import numpy as np
import pytest
import tifffile

from careamics import CAREamist, Configuration, save_configuration
from careamics.config.support import SupportedAlgorithm, SupportedData


def test_no_parameters():
    """Test that CAREamics cannot be instantiated without parameters."""
    with pytest.raises(TypeError):
        CAREamist()


def test_minimum_configuration_via_object(tmp_path, minimum_configuration):
    """Test that CAREamics can be instantiated with a minimum configuration object."""
    # create configuration
    config = Configuration(**minimum_configuration)

    # instantiate CAREamist
    CAREamist(source=config, work_dir=tmp_path)


def test_minimum_configuration_via_path(tmp_path, minimum_configuration):
    """Test that CAREamics can be instantiated with a path to a minimum
    configuration.
    """
    # create configuration
    config = Configuration(**minimum_configuration)
    path_to_config = save_configuration(config, tmp_path)

    # instantiate CAREamist
    CAREamist(source=path_to_config)


def test_train_error_target_unsupervised_algorithm(tmp_path, minimum_configuration):
    """Test that an error is raised when a target is provided for N2V."""
    # create configuration
    config = Configuration(**minimum_configuration)
    config.algorithm_config.algorithm = SupportedAlgorithm.N2V.value

    # train error with Paths
    config.data_config.data_type = SupportedData.TIFF.value
    careamics = CAREamist(source=config, work_dir=tmp_path)
    with pytest.raises(ValueError):
        careamics.train(
            train_source=tmp_path,
            train_target=tmp_path,
        )

    # train error with strings
    with pytest.raises(ValueError):
        careamics.train(
            train_source=str(tmp_path),
            train_target=str(tmp_path),
        )

    # train error with arrays
    config.data_config.data_type = SupportedData.ARRAY.value
    careamics = CAREamist(source=config, work_dir=tmp_path)
    with pytest.raises(ValueError):
        careamics.train(
            train_source=np.ones((32, 32)),
            train_target=np.ones((32, 32)),
        )


def test_train_single_array_no_val(tmp_path, minimum_configuration):
    """Test that CAREamics can be trained with arrays."""
    # training data
    train_array = np.ones((32, 32))

    # create configuration
    config = Configuration(**minimum_configuration)
    config.training_config.num_epochs = 1
    config.data_config.axes = "YX"
    config.data_config.batch_size = 2
    config.data_config.data_type = SupportedData.ARRAY.value
    config.data_config.patch_size = (8, 8)

    # instantiate CAREamist
    careamist = CAREamist(source=config, work_dir=tmp_path)

    # train CAREamist
    careamist.train(train_source=train_array)

    # check that it trained
    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()


def test_train_array(tmp_path, minimum_configuration):
    """Test that CAREamics can be trained with arrays."""
    # training data
    train_array = np.ones((32, 32))
    val_array = np.ones((32, 32))

    # create configuration
    config = Configuration(**minimum_configuration)
    config.training_config.num_epochs = 1
    config.data_config.axes = "YX"
    config.data_config.batch_size = 2
    config.data_config.data_type = SupportedData.ARRAY.value
    config.data_config.patch_size = (8, 8)

    # instantiate CAREamist
    careamist = CAREamist(source=config, work_dir=tmp_path)

    # train CAREamist
    careamist.train(train_source=train_array, val_source=val_array)

    # check that it trained
    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()


def test_train_tiff_files_in_memory_no_val(tmp_path, minimum_configuration):
    """Test that CAREamics can be trained with tiff files in memory."""
    # training data
    train_array = np.ones((32, 32))

    # save files
    train_file = tmp_path / "train.tiff"
    tifffile.imwrite(train_file, train_array)

    # create configuration
    config = Configuration(**minimum_configuration)
    config.training_config.num_epochs = 1
    config.data_config.axes = "YX"
    config.data_config.batch_size = 2
    config.data_config.data_type = SupportedData.TIFF.value
    config.data_config.patch_size = (8, 8)

    # instantiate CAREamist
    careamist = CAREamist(source=config, work_dir=tmp_path)

    # train CAREamist
    careamist.train(train_source=train_file)

    # check that it trained
    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()


def test_train_tiff_files_in_memory(tmp_path, minimum_configuration):
    """Test that CAREamics can be trained with tiff files in memory."""
    # training data
    train_array = np.ones((32, 32))
    val_array = np.ones((32, 32))

    # save files
    train_file = tmp_path / "train.tiff"
    tifffile.imwrite(train_file, train_array)

    val_file = tmp_path / "val.tiff"
    tifffile.imwrite(val_file, val_array)

    # create configuration
    config = Configuration(**minimum_configuration)
    config.training_config.num_epochs = 1
    config.data_config.axes = "YX"
    config.data_config.batch_size = 2
    config.data_config.data_type = SupportedData.TIFF.value
    config.data_config.patch_size = (8, 8)

    # instantiate CAREamist
    careamist = CAREamist(source=config, work_dir=tmp_path)

    # train CAREamist
    careamist.train(train_source=train_file, val_source=val_file)

    # check that it trained
    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()


def test_train_tiff_files(tmp_path, minimum_configuration):
    """Test that CAREamics can be trained with tiff files by deactivating
    the in memory dataset.
    """
    # training data
    train_array = np.ones((32, 32))
    val_array = np.ones((32, 32))

    # save files
    train_file = tmp_path / "train.tiff"
    tifffile.imwrite(train_file, train_array)

    val_file = tmp_path / "val.tiff"
    tifffile.imwrite(val_file, val_array)

    # create configuration
    config = Configuration(**minimum_configuration)
    config.training_config.num_epochs = 1
    config.data_config.axes = "YX"
    config.data_config.batch_size = 2
    config.data_config.data_type = SupportedData.TIFF.value
    config.data_config.patch_size = (8, 8)

    # instantiate CAREamist
    careamist = CAREamist(source=config, work_dir=tmp_path)

    # train CAREamist
    careamist.train(train_source=train_file, val_source=val_file, use_in_memory=False)

    # check that it trained
    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()


def test_train_array_supervised(tmp_path, supervised_configuration):
    """Test that CAREamics can be trained with arrays."""
    # training data
    train_array = np.ones((32, 32))
    val_array = np.ones((32, 32))
    train_target = np.ones((32, 32))
    val_target = np.ones((32, 32))

    # create configuration
    config = Configuration(**supervised_configuration)
    config.training_config.num_epochs = 1
    config.data_config.axes = "YX"
    config.data_config.batch_size = 2
    config.data_config.data_type = SupportedData.ARRAY.value
    config.data_config.patch_size = (8, 8)

    # instantiate CAREamist
    careamist = CAREamist(source=config, work_dir=tmp_path)

    # train CAREamist
    careamist.train(
        train_source=train_array,
        val_source=val_array,
        train_target=train_target,
        val_target=val_target,
    )

    # check that it trained
    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()


def test_train_tiff_files_in_memory_supervised(tmp_path, supervised_configuration):
    """Test that CAREamics can be trained with tiff files in memory."""
    # training data
    train_array = np.ones((32, 32))
    val_array = np.ones((32, 32))
    train_target = np.ones((32, 32))
    val_target = np.ones((32, 32))

    # save files
    images = tmp_path / "images"
    images.mkdir()
    train_file = images / "train.tiff"
    tifffile.imwrite(train_file, train_array)

    val_file = tmp_path / "images" / "val.tiff"
    tifffile.imwrite(val_file, val_array)

    targets = tmp_path / "targets"
    targets.mkdir()
    train_target_file = targets / "train.tiff"
    tifffile.imwrite(train_target_file, train_target)

    val_target_file = targets / "val.tiff"
    tifffile.imwrite(val_target_file, val_target)

    # create configuration
    config = Configuration(**supervised_configuration)
    config.training_config.num_epochs = 1
    config.data_config.axes = "YX"
    config.data_config.batch_size = 2
    config.data_config.data_type = SupportedData.TIFF.value
    config.data_config.patch_size = (8, 8)

    # instantiate CAREamist
    careamist = CAREamist(source=config, work_dir=tmp_path)

    # train CAREamist
    careamist.train(
        train_source=train_file,
        val_source=val_file,
        train_target=train_target_file,
        val_target=val_target_file,
    )

    # check that it trained
    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()


def test_train_tiff_files_supervised(tmp_path, supervised_configuration):
    """Test that CAREamics can be trained with tiff files by deactivating
    the in memory dataset.
    """
    # training data
    train_array = np.ones((32, 32))
    val_array = np.ones((32, 32))
    train_target = np.ones((32, 32))
    val_target = np.ones((32, 32))

    # save files
    images = tmp_path / "images"
    images.mkdir()
    train_file = images / "train.tiff"
    tifffile.imwrite(train_file, train_array)

    val_file = tmp_path / "images" / "val.tiff"
    tifffile.imwrite(val_file, val_array)

    targets = tmp_path / "targets"
    targets.mkdir()
    train_target_file = targets / "train.tiff"
    tifffile.imwrite(train_target_file, train_target)

    val_target_file = targets / "val.tiff"
    tifffile.imwrite(val_target_file, val_target)

    # create configuration
    config = Configuration(**supervised_configuration)
    config.training_config.num_epochs = 1
    config.data_config.axes = "YX"
    config.data_config.batch_size = 2
    config.data_config.data_type = SupportedData.TIFF.value
    config.data_config.patch_size = (8, 8)

    # instantiate CAREamist
    careamist = CAREamist(source=config, work_dir=tmp_path)

    # train CAREamist
    careamist.train(
        train_source=train_file,
        val_source=val_file,
        train_target=train_target_file,
        val_target=val_target_file,
        use_in_memory=False,
    )

    # check that it trained
    assert Path(tmp_path / "checkpoints" / "last.ckpt").exists()


@pytest.mark.parametrize("batch_size", [1, 2])
def test_predict_array(tmp_path, minimum_configuration, batch_size):
    """Test that CAREamics can predict with arrays."""
    # training data
    train_array = np.random.rand(32, 32)

    # create configuration
    config = Configuration(**minimum_configuration)
    config.training_config.num_epochs = 1
    config.data_config.axes = "YX"
    config.data_config.batch_size = 2
    config.data_config.data_type = SupportedData.ARRAY.value
    config.data_config.patch_size = (8, 8)

    # instantiate CAREamist
    careamist = CAREamist(source=config, work_dir=tmp_path)

    # train CAREamist
    careamist.train(train_source=train_array)

    # predict CAREamist
    predicted = careamist.predict(
        train_array, batch_size=batch_size, tile_overlap=(4, 4)
    )

    # check thatmean/std were set properly
    assert careamist.cfg.data_config.mean is not None
    assert careamist.cfg.data_config.std is not None
    assert careamist.cfg.data_config.mean == train_array.mean()
    assert careamist.cfg.data_config.std == train_array.std()
    # check prediction and its shape@pytest.mark.parametrize("batch_size", [1, 2])

    assert predicted is not None
    assert predicted.squeeze().shape == train_array.shape


@pytest.mark.parametrize("batch_size", [1, 2])
def test_predict_path(tmp_path, minimum_configuration, batch_size):
    """Test that CAREamics can predict with tiff files."""
    # training data
    train_array = np.random.rand(32, 32)

    # save files
    train_file = tmp_path / "train.tiff"
    tifffile.imwrite(train_file, train_array)

    # create configuration
    config = Configuration(**minimum_configuration)
    config.training_config.num_epochs = 1
    config.data_config.axes = "YX"
    config.data_config.batch_size = 2
    config.data_config.data_type = SupportedData.TIFF.value
    config.data_config.patch_size = (8, 8)

    # instantiate CAREamist
    careamist = CAREamist(source=config, work_dir=tmp_path)

    # train CAREamist
    careamist.train(train_source=train_file)

    # predict CAREamist
    predicted = careamist.predict(
        train_file, batch_size=batch_size, tile_overlap=(4, 4)
    )

    # check that it predicted
    assert predicted is not None
    assert predicted.squeeze().shape == train_array.shape


def test_predict_pretrained(tmp_path, pre_trained):
    """Test that CAREamics can be instantiated with a pre-trained network and predict
    on an array."""
    # training data
    train_array = np.ones((32, 32))

    # instantiate CAREamist
    careamist = CAREamist(source=pre_trained, work_dir=tmp_path)
    assert careamist.cfg.data_config.mean is not None
    assert careamist.cfg.data_config.std is not None

    # predict
    predicted = careamist.predict(train_array, tile_overlap=(4, 4))

    # check that it predicted
    assert predicted is not None
    assert predicted.squeeze().shape == train_array.shape
