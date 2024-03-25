import numpy as np
import pytest
import tifffile

from careamics import CAREamist, Configuration, save_configuration
from careamics.config.support import SupportedAlgorithm, SupportedData


def test_no_parameters():
    """Test that CAREamics cannot be instantiated without parameters."""
    with pytest.raises(ValueError):
        CAREamist()


def test_minimum_configuration_via_object(minimum_configuration):
    """Test that CAREamics can be instantiated with a minimum configuration object."""
    # create configuration
    config = Configuration(**minimum_configuration)

    # instantiate CAREamist
    CAREamist(configuration=config)


def test_minimum_configuration_via_path(tmp_path, minimum_configuration):
    """Test that CAREamics can be instantiated with a path to a minimum
    configuration.
    """
    # create configuration
    config = Configuration(**minimum_configuration)
    path_to_config = save_configuration(config, tmp_path)

    # instantiate CAREamist
    CAREamist(path_to_config=path_to_config)


def test_train_error_target_unsupervised_algorithm(tmp_path, minimum_configuration):
    """Test that an error is raised when a target is provided for an unsupervised
    algorithm.
    """
    # create configuration
    config = Configuration(**minimum_configuration)
    config.algorithm.algorithm = SupportedAlgorithm.N2V.value

    # train error with Paths
    config.data.data_type = SupportedData.TIFF.value
    careamics = CAREamist(configuration=config)
    with pytest.raises(ValueError):
        careamics._train_on_path(
            path_to_train_data=tmp_path,
            path_to_train_target=tmp_path,
        )

    # train error with strings
    with pytest.raises(ValueError):
        careamics._train_on_path(
            path_to_train_data=str(tmp_path),
            path_to_train_target=str(tmp_path),
        )

    # train error with arrays
    config.data.data_type = SupportedData.ARRAY.value
    careamics = CAREamist(configuration=config)
    with pytest.raises(ValueError):
        careamics._train_on_array(
            train_data=np.ones((32, 32)),
            train_target=np.ones((32, 32)),
        )


def test_train_array(minimum_configuration):
    """Test that CAREamics can be trained with arrays."""
    # training data
    train_array = np.ones((32, 32))
    val_array = np.ones((32, 32))

    # create configuration
    config = Configuration(**minimum_configuration)
    config.training.num_epochs = 1
    config.training.batch_size = 2
    config.data.axes = "YX"
    config.data.data_type = SupportedData.ARRAY.value
    config.data.patch_size = (8, 8)

    # instantiate CAREamist
    careamist = CAREamist(configuration=config)

    # train CAREamist
    careamist._train_on_array(train_array, val_array)

    # check that it recorded mean and std
    assert careamist.cfg.data.mean is not None
    assert careamist.cfg.data.std is not None
    # TODO somethign to check that it trained, maybe through callback


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
    config.training.num_epochs = 1
    config.training.batch_size = 2
    config.data.axes = "YX"
    config.data.data_type = SupportedData.TIFF.value
    config.data.patch_size = (8, 8)

    # instantiate CAREamist
    careamist = CAREamist(configuration=config)

    # train CAREamist
    careamist._train_on_path(train_file, val_file)

    # check that it recorded mean and std
    assert careamist.cfg.data.mean is not None
    assert careamist.cfg.data.std is not None
    # TODO somethign to check that it trained, maybe through callback


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
    config.training.num_epochs = 1
    config.training.batch_size = 2
    config.data.axes = "YX"
    config.data.data_type = SupportedData.TIFF.value
    config.data.patch_size = (8, 8)

    # instantiate CAREamist
    careamist = CAREamist(configuration=config)

    # train CAREamist
    careamist._train_on_path(train_file, val_file, use_in_memory=False)

    # check that it recorded mean and std
    assert careamist.cfg.data.mean is not None
    assert careamist.cfg.data.std is not None
    # TODO somethign to check that it trained, maybe through callback
