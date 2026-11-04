import pytest

from careamics.config.architectures import LVAEConfig
from careamics.config.support import SupportedActivation

pytestmark = pytest.mark.lvae


def test_instantiation():
    """Test that LVAEModel can be instantiated."""
    model_params = {
        "architecture": "LVAE",
    }

    # instantiate model
    LVAEConfig(**model_params)


def test_architecture_missing():
    """Test that LVAEModel requires architecture."""
    model_params = {
        "input_shape": 64,
    }

    with pytest.raises(ValueError):
        LVAEConfig(**model_params)


@pytest.mark.parametrize("n_filters", [8, 16, 32, 96, 128])
def test_n_filters(n_filters: int):
    """Test that LVAEModel accepts num_channels_init as an even number and
    minimum 8."""
    model_params = {
        "architecture": "LVAE",
        "n_filters": n_filters,
    }

    # instantiate model
    LVAEConfig(**model_params)


@pytest.mark.parametrize("n_filters", [2, 17, 127])
def test_wrong_num_filters(n_filters: int):
    """Test that wrong num_channels_init causes an error."""
    model_params = {"architecture": "LVAE", "n_filters": n_filters}
    with pytest.raises(ValueError):
        LVAEConfig(**model_params)


def test_activations():
    """Test that LVAEModel accepts all activations."""
    for act in SupportedActivation:
        model_params = {
            "architecture": "LVAE",
            "nonlinearity": act.value,
        }

        # instantiate model
        LVAEConfig(**model_params)


def test_all_activations_are_supported():
    """Test that all activations defined in the Literal are supported."""
    # list of supported activations
    activations = list(SupportedActivation)

    # Algorithm json schema
    schema = LVAEConfig.model_json_schema()

    # check that all activations are supported
    for act in schema["properties"]["nonlinearity"]["enum"]:
        assert act in activations


def test_activation_wrong_values():
    """Test that wrong values are not accepted."""
    model_params = {
        "architecture": "LVAE",
        "nonlinearity": "wrong",
    }

    with pytest.raises(ValueError):
        LVAEConfig(**model_params)


def test_parameters_wrong_values_by_assigment():
    """Test that wrong values are not accepted through assignment."""
    model_params = {
        "architecture": "LVAE",
        "z_dims": (128, 128, 128),
        "multiscale_count": 2,
        "n_filters": 32,
    }
    model = LVAEConfig(**model_params)

    # z_dims
    model.z_dims = model_params["z_dims"]
    with pytest.raises(ValueError):
        model.depth = -1

    # number of channels
    model.n_filters = model_params["n_filters"]
    with pytest.raises(ValueError):
        model.n_filters = 2
