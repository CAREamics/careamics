from pathlib import Path

import pytest

from careamics.config import (
    Configuration,
    load_configuration,
    save_configuration,
)


@pytest.mark.parametrize("name", ["Sn4K3", "C4_M e-L"])
def test_config_valid_names(minimum_configuration: dict, name: str):
    """Test valid names (letters, numbers, spaces, dashes and underscores)."""
    minimum_configuration["experiment_name"] = name
    myconf = Configuration(**minimum_configuration)
    assert myconf.experiment_name == name


@pytest.mark.parametrize("name", ["", "   ", "#", "/", "^", "%", ",", ".", "a=b"])
def test_config_invalid_names(minimum_configuration: dict, name: str):
    """Test that invalid names raise an error."""
    minimum_configuration["experiment_name"] = name
    with pytest.raises(ValueError):
        Configuration(**minimum_configuration)


@pytest.mark.parametrize("path", ["", "tmp"])
def test_config_valid_working_directory(
    tmp_path: Path, minimum_configuration: dict, path: str
):
    """Test valid working directory.

    A valid working directory exists or its direct parent exists.
    """
    path = tmp_path / path
    minimum_configuration["working_directory"] = str(path)
    myconf = Configuration(**minimum_configuration)
    assert myconf.working_directory == path


def test_config_invalid_working_directory(tmp_path: Path, minimum_configuration: dict):
    """Test that invalid working directory raise an error.

    Since its parent does not exist, this case is invalid.
    """
    path = tmp_path / "tmp" / "tmp"
    minimum_configuration["working_directory"] = str(path)
    with pytest.raises(ValueError):
        Configuration(**minimum_configuration)

    path = tmp_path / "tmp.txt"
    path.touch()
    minimum_configuration["working_directory"] = str(path)
    with pytest.raises(ValueError):
        Configuration(**minimum_configuration)


def test_3D_algorithm_and_data_compatibility(minimum_configuration: dict):
    """Test that errors are raised if algorithm `is_3D` and data axes are 
    incompatible.
    """
    # 3D but no Z in axes
    minimum_configuration["algorithm"]["model"]["conv_dims"] = 3
    with pytest.raises(ValueError):
        Configuration(**minimum_configuration)

    # 2D but Z in axes
    minimum_configuration["algorithm"]["model"]["conv_dims"] = 2
    minimum_configuration["data"]["axes"] = "ZYX"
    with pytest.raises(ValueError):
        Configuration(**minimum_configuration)


def test_set_3D(minimum_configuration: dict):
    """Test the set 3D method."""
    conf = Configuration(**minimum_configuration)

    # set to 3D
    conf.set_3D(True, "ZYX")

    # set to 2D
    conf.set_3D(False, "SYX")

    # fails if they are not compatible
    with pytest.raises(ValueError):
        conf.set_3D(True, "SYX")

    with pytest.raises(ValueError):
        conf.set_3D(False, "ZYX")



def test_config_to_yaml(tmp_path: Path, minimum_configuration: dict):
    """Test that we can export a config to yaml and load it back"""

    # test that we can instantiate a config
    myconf = Configuration(**minimum_configuration)

    # export to yaml
    yaml_path = save_configuration(myconf, tmp_path)
    assert yaml_path.exists()

    # load from yaml
    my_other_conf = load_configuration(yaml_path)
    assert my_other_conf == myconf


def test_config_to_yaml_wrong_path(tmp_path: Path, minimum_configuration: dict):
    """Test that an error is raised when the path is not a directory and not a .yml"""

    # test that we can instantiate a config
    myconf = Configuration(**minimum_configuration)

    # export to yaml
    yaml_path = tmp_path / "tmp.txt"
    with pytest.raises(ValueError):
        save_configuration(myconf, yaml_path)

    # existing file
    yaml_path.touch()
    with pytest.raises(ValueError):
        save_configuration(myconf, yaml_path)
