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


@pytest.mark.parametrize("model_path", ["model.pth", "tmp/model.pth"])
def test_config_valid_model(tmp_path: Path, complete_config: dict, model_path: str):
    """Test valid model path."""
    path = tmp_path / model_path
    path.parent.mkdir(exist_ok=True, parents=True)
    path.touch()

    complete_config["working_directory"] = tmp_path
    complete_config["trained_model"] = model_path

    myconf = Configuration(**complete_config)
    assert myconf.trained_model == model_path


def test_config_valid_model_absolute(tmp_path: Path, complete_config: dict):
    """Test valid model path."""
    path = tmp_path / "tmp1/tmp2/model.pth"
    path.parent.mkdir(exist_ok=True, parents=True)
    path.touch()

    complete_config["working_directory"] = tmp_path
    complete_config["trained_model"] = str(path.absolute())

    myconf = Configuration(**complete_config)
    assert myconf.trained_model == complete_config["trained_model"]


@pytest.mark.parametrize("model_path", ["model", "tmp/model"])
def test_config_invalid_model(tmp_path: Path, complete_config: dict, model_path: str):
    """Test that invalid model path raise an error."""
    complete_config["working_directory"] = tmp_path
    complete_config["trained_model"] = model_path
    with pytest.raises(ValueError):
        Configuration(**complete_config)


def test_minimum_config(minimum_config: dict):
    """Test that we can instantiate a minimum config."""
    dictionary = Configuration(**minimum_config).model_dump()
    assert dictionary == minimum_config


def test_complete_config(complete_config: dict):
    """Test that we can instantiate a minimum config."""
    dictionary = Configuration(**complete_config).model_dump()
    assert dictionary == complete_config


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


def test_get_stage(complete_config: dict):
    """Test that we can get the configuration for a specific stage."""

    # test that we can instantiate a config
    myconf = Configuration(**complete_config)

    # get training config
    training_config = myconf.get_stage_config("training")
    assert training_config == myconf.training

    # get prediction config
    prediction_config = myconf.get_stage_config("prediction")
    assert prediction_config == myconf.prediction

    with pytest.raises(ValueError):
        myconf.get_stage_config("not_a_stage")
