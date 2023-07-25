import copy
from pathlib import Path

import pytest

from careamics_restoration.config import (
    Configuration,
    load_configuration,
    save_configuration,
)


@pytest.mark.parametrize("name", ["Sn4K3", "C4_M e-L"])
def test_config_valid_names(minimum_config: dict, name: str):
    """Test valid names (letters, numbers, spaces, dashes and underscores)."""
    minimum_config["experiment_name"] = name
    myconf = Configuration(**minimum_config)
    assert myconf.experiment_name == name


@pytest.mark.parametrize("name", ["", "   ", "#", "/", "^", "%", ",", ".", "a=b"])
def test_config_invalid_names(minimum_config: dict, name: str):
    """Test that invalid names raise an error."""
    minimum_config["experiment_name"] = name
    with pytest.raises(ValueError):
        Configuration(**minimum_config)


@pytest.mark.parametrize("path", ["", "tmp"])
def test_config_valid_working_directory(
    tmp_path: Path, minimum_config: dict, path: str
):
    """Test valid working directory.

    A valid working directory exists or its direct parent exists.
    """
    path = tmp_path / path
    minimum_config["working_directory"] = str(path)
    myconf = Configuration(**minimum_config)
    assert myconf.working_directory == path


def test_config_invalid_working_directory(tmp_path: Path, minimum_config: dict):
    """Test that invalid working directory raise an error.

    Since its parent does not exist, this case is invalid.
    """
    path = tmp_path / "tmp/tmp"
    minimum_config["working_directory"] = str(path)
    with pytest.raises(ValueError):
        Configuration(**minimum_config)


def test_3D_algorithm_and_data_compatibility(minimum_config: dict):
    """Test that errors are raised if algithm `is_3D` and data axes are incompatible."""
    # 3D but no Z in axes
    minimum_config["algorithm"]["is_3D"] = True
    with pytest.raises(ValueError):
        Configuration(**minimum_config)

    # 2D but Z in axes
    minimum_config["algorithm"]["is_3D"] = False
    minimum_config["data"]["axes"] = "ZYX"
    with pytest.raises(ValueError):
        Configuration(**minimum_config)


def test_set_3D(minimum_config: dict):
    """Test the set 3D method."""
    conf = Configuration(**minimum_config)

    # set to 3D
    conf.set_3D(True, "ZYX")

    # set to 2D
    conf.set_3D(False, "SYX")

    # fails if they are not compatible
    with pytest.raises(ValueError):
        conf.set_3D(True, "SYX")

    with pytest.raises(ValueError):
        conf.set_3D(False, "ZYX")


def test_at_least_one_of_training_or_prediction(complete_config: dict):
    """Test that at least one of training or prediction is specified."""
    test_config = copy.deepcopy(complete_config)

    # remove training and prediction
    test_config.pop("training")
    test_config.pop("prediction")

    # test that config is invalid
    with pytest.raises(ValueError):
        Configuration(**test_config)

    # test that config is valid if we add training
    test_config["training"] = copy.deepcopy(complete_config["training"])
    config_train = Configuration(**test_config)
    assert config_train.training.model_dump() == test_config["training"]

    # remove training
    test_config.pop("training")

    # test that config is valid if we add prediction
    test_config["prediction"] = copy.deepcopy(complete_config["prediction"])
    config_pred = Configuration(**test_config)
    assert config_pred.prediction.model_dump() == test_config["prediction"]


def test_wrong_values_by_assignment(complete_config: dict):
    """Test that wrong values raise an error when assigned."""
    config = Configuration(**complete_config)

    # experiment name
    config.experiment_name = "My name is Inigo Montoya"
    with pytest.raises(ValueError):
        config.experiment_name = "¯\\_(ツ)_/¯"

    # working directory
    config.working_directory = complete_config["working_directory"]
    with pytest.raises(ValueError):
        config.working_directory = "o/o"

    # data
    config.data = complete_config["data"]
    with pytest.raises(ValueError):
        config.data = "I am not a data model"

    # algorithm
    config.algorithm = complete_config["algorithm"]
    with pytest.raises(ValueError):
        config.algorithm = None

    # training
    config.training = complete_config["training"]
    with pytest.raises(ValueError):
        config.training = "Hubert Blaine Wolfeschlegelsteinhausenbergerdorff Sr."

    # prediction
    config.prediction = complete_config["prediction"]
    with pytest.raises(ValueError):
        config.prediction = "You can call me Giorgio"

    # Test that the model validators are also run after assigment
    config.prediction = None
    with pytest.raises(ValueError):
        config.training = None

    # TODO Because algorithm is a sub-model of Configuration, and the validation is
    # done at the level of the Configuration, this does not cause any error, although
    # it should.
    config.algorithm.is_3D = True


def test_minimum_config(minimum_config: dict):
    """Test that we can instantiate a minimum config."""
    dictionary = Configuration(**minimum_config).model_dump()
    assert dictionary == minimum_config


def test_complete_config(complete_config: dict):
    """Test that we can instantiate a minimum config."""
    dictionary = Configuration(**complete_config).model_dump()
    assert dictionary == complete_config


def test_config_to_dict_with_default_optionals(complete_config: dict):
    """Test that the exclude optional options in model dump gives a full configuration,
    including the default optional values.

    Note that None values are always excluded.
    """
    # Algorithm default optional parameters
    complete_config["algorithm"]["masking_strategy"] = "default"
    complete_config["algorithm"]["masked_pixel_percentage"] = 0.2
    complete_config["algorithm"]["model_parameters"] = {
        "depth": 2,
        "num_channels_init": 96,
    }

    # Training default optional parameters
    complete_config["training"]["optimizer"]["parameters"] = {}
    complete_config["training"]["lr_scheduler"]["parameters"] = {}
    complete_config["training"]["use_wandb"] = True
    complete_config["training"]["num_workers"] = 0
    complete_config["training"]["amp"] = {
        "use": True,
        "init_scale": 1024,
    }

    # instantiate config
    myconf = Configuration(**complete_config)
    assert myconf.model_dump(exclude_optionals=False) == complete_config


def test_config_to_yaml(tmp_path: Path, minimum_config: dict):
    """Test that we can export a config to yaml and load it back"""

    # test that we can instantiate a config
    myconf = Configuration(**minimum_config)

    # export to yaml
    yaml_path = save_configuration(myconf, tmp_path)
    assert yaml_path.exists()

    # load from yaml
    my_other_conf = load_configuration(yaml_path)
    assert my_other_conf == myconf
