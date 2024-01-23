import pytest

from careamics.config.architectures import UNet


@pytest.mark.parametrize("num_channels_init", [8, 16, 32, 96, 128])
def test_num_channels_init(
    num_channels_init: int
):
    """Test that UNet accepts num_channels_init as an even number and
    minimum 8."""
    model_params = {
        "num_channels_init": num_channels_init
    }

    # instantiate model
    UNet(**model_params)


@pytest.mark.parametrize("num_channels_init", [2, 17, 127])
def test_wrong_num_channels_init(
    num_channels_init: int
):
    """Test that wrong num_channels_init causes an error."""
    model_params = {
        "num_channels_init": num_channels_init
    }

    with pytest.raises(ValueError):
        UNet(**model_params)


def test_parameters_wrong_values_by_assigment():
    """Test that wrong values are not accepted through assignment."""
    model_params = {
        "num_channels_init": 16,
        "depth": 2
    }
    model = UNet(**model_params)

    # depth
    model.depth = model_params["depth"]
    with pytest.raises(ValueError):
        model.depth = -1

    # number of channels
    model.num_channels_init = model_params["num_channels_init"]
    with pytest.raises(ValueError):
        model.num_channels_init = 2
