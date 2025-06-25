import pytest
from pydantic import ValidationError

from careamics.config import Configuration


@pytest.mark.parametrize("name", ["Sn4K3", "C4_M e-L"])
def test_valid_names(minimum_supervised_configuration: dict, name: str):
    """Test valid names (letters, numbers, spaces, dashes and underscores)."""
    minimum_supervised_configuration["experiment_name"] = name
    myconf = Configuration(**minimum_supervised_configuration)
    assert myconf.experiment_name == name


@pytest.mark.parametrize("name", ["", "   ", "#", "/", "^", "%", ",", ".", "a=b"])
def test_invalid_names(minimum_supervised_configuration: dict, name: str):
    """Test that invalid names raise an error."""
    minimum_supervised_configuration["experiment_name"] = name
    with pytest.raises(ValueError):
        Configuration(**minimum_supervised_configuration)


def test_3D_algorithm_and_data_compatibility(minimum_supervised_configuration: dict):
    """Test that errors are raised if algorithm `is_3D` and data axes are
    incompatible.
    """
    # 3D but no Z in axes
    minimum_supervised_configuration["algorithm_config"]["model"]["conv_dims"] = 3
    config = Configuration(**minimum_supervised_configuration)
    assert config.algorithm_config.model.conv_dims == 2

    # 2D but Z in axes
    minimum_supervised_configuration["algorithm_config"]["model"]["conv_dims"] = 2
    minimum_supervised_configuration["data_config"]["axes"] = "ZYX"
    minimum_supervised_configuration["data_config"]["patch_size"] = [64, 64, 64]
    config = Configuration(**minimum_supervised_configuration)
    assert config.algorithm_config.model.conv_dims == 3


def test_set_3D(minimum_supervised_configuration: dict):
    """Test the set 3D method."""
    conf = Configuration(**minimum_supervised_configuration)

    # set to 3D
    conf.set_3D(True, "ZYX", [64, 64, 64])
    assert conf.data_config.axes == "ZYX"
    assert conf.data_config.patch_size == [64, 64, 64]
    assert conf.algorithm_config.model.conv_dims == 3

    # set to 2D
    conf.set_3D(False, "SYX", [64, 64])
    assert conf.data_config.axes == "SYX"
    assert conf.data_config.patch_size == [64, 64]
    assert conf.algorithm_config.model.conv_dims == 2


def test_validate_n2v_mask_pixel_perc(minimum_n2v_configuration):

    minimum_n2v_configuration["data_config"]["patch_size"] = [16, 16]
    minimum_n2v_configuration["algorithm_config"]["n2v_config"][
        "masked_pixel_percentage"
    ] = 0.2

    with pytest.raises(ValidationError):
        Configuration(**minimum_n2v_configuration)
