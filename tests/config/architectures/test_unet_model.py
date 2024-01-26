import pytest

from careamics.config.architectures import UNetModel


def test_instantiation():
    """Test that UNet can be instantiated."""
    model_params = {
        "architecture": "UNet",
        "conv_dim": 2,
        "num_channels_init": 16,
    }

    # instantiate model
    UNetModel(**model_params)


def test_architecture_missing():
    """Test that UNet requires architecture."""
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
    """Test that UNet accepts num_channels_init as an even number and
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
        "num_channels_init": num_channels_init
    }

    with pytest.raises(ValueError):
        UNetModel(**model_params)


@pytest.mark.parametrize("acitvation", ["none", "sigmoid", "softmax"])
def test_activation(
    acitvation: str
):
    """Test that UNet accepts activation as none, sigmoid or softmax."""
    model_params = {
        "architecture": "UNet",
        "num_channels_init": 16,
        "final_activation": acitvation
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
