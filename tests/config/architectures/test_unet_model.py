import pytest

from careamics.config.architectures import UNetModel
from careamics.config.support import SupportedArchitecture, SupportedActivation


def test_instantiation():
    """Test that UNetModel can be instantiated."""
    model_params = {
        "architecture": "UNet",
        "conv_dim": 2,
        "num_channels_init": 16,
    }

    # instantiate model
    UNetModel(**model_params)


def test_architecture_missing():
    """Test that UNetModel requires architecture."""
    model_params = {
        "depth": 2,
        "num_channels_init": 16,
    }

    with pytest.raises(ValueError):
        UNetModel(**model_params)


@pytest.mark.parametrize("num_channels_init", [8, 16, 32, 96, 128])
def test_num_channels_init(
    num_channels_init: int
):
    """Test that UNetModel accepts num_channels_init as an even number and
    minimum 8."""
    model_params = {
        "architecture": "UNet",
        "num_channels_init": num_channels_init
    }

    # instantiate model
    UNetModel(**model_params)


@pytest.mark.parametrize("num_channels_init", [2, 17, 127])
def test_wrong_num_channels_init(
    num_channels_init: int
):
    """Test that wrong num_channels_init causes an error."""
    model_params = {
        "architecture": "UNet",
        "num_channels_init": num_channels_init
    }

    with pytest.raises(ValueError):
        UNetModel(**model_params)


def test_activations():
    """Test that UNetModel accepts all activations."""
    for act in SupportedActivation:    
        model_params = {
            "architecture": "UNet",
            "num_channels_init": 16,
            "final_activation": act.value
        }

        # instantiate model
        UNetModel(**model_params)


def test_activation_wrong_values():
    """Test that wrong values are not accepted."""
    model_params = {
        "architecture": "UNet",
        "num_channels_init": 16,
        "final_activation": "wrong"
    }

    with pytest.raises(ValueError):
        UNetModel(**model_params)


def test_parameters_wrong_values_by_assigment():
    """Test that wrong values are not accepted through assignment."""
    model_params = {
        "architecture": "UNet",
        "num_channels_init": 16,
        "depth": 2
    }
    model = UNetModel(**model_params)

    # depth
    model.depth = model_params["depth"]
    with pytest.raises(ValueError):
        model.depth = -1

    # number of channels
    model.num_channels_init = model_params["num_channels_init"]
    with pytest.raises(ValueError):
        model.num_channels_init = 2


def test_model_dump():
    """Test that default values are excluded from model dump, and that the enum 
    are replaced by their values."""
    model_params = {
        "architecture": "UNet",
        "num_channels_init": 16, # non-default value
        "final_activation": "ReLU", # non-default value
    }
    model = UNetModel(**model_params)

    # dump model
    model_dict = model.model_dump(exclude_defaults=True)

    # check that default values are excluded except the architecture
    assert "architecture" in model_dict
    assert len(model_dict) == 3

    # check that enum are not passed as enums but as their values
    assert not isinstance(model_dict["architecture"], SupportedArchitecture)
    assert not isinstance(model_dict["final_activation"], SupportedActivation)

    # check that we get all the optional values with the exclude_defaults flag
    model_dict = model.model_dump(exclude_defaults=False)
    assert len(model_dict) == len(dict(model))
