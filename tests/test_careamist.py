import pytest

import numpy as np

from careamics import CAREamist, Configuration, save_configuration
from careamics.config.support import SupportedData


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



def test_train_array(minimum_configuration):
    """Test that CAREamics can be trained with arrays."""
    # training data
    train_array = np.ones((32, 32))
    val_array = np.ones((32, 32))

    # create configuration
    config = Configuration(**minimum_configuration)
    config.training.num_epochs = 1
    config.training.batch_size = 1
    config.data.axes = "YX"
    config.data.data_type = SupportedData.ARRAY.value
    config.data.patch_size = (8, 8)

    # instantiate CAREamist
    careamist = CAREamist(configuration=config)

    # train CAREamist
    careamist.train(train_array, val_array)