import pytest
from pydantic.error_wrappers import ValidationError

from careamics_restoration.config.algorithm import Algorithm


def test_algorithm(minimum_config):
    """Test that we can instantiate a config with a valid algorithm."""
    algorithm_config = minimum_config["algorithm"]
    _ = Algorithm(**algorithm_config)


def test_wrong_loss_value(minimum_config):
    """Test that we cannot instantiate a config with wrong loss value."""
    algorithm_config = minimum_config["algorithm"]

    # wrong entry
    algorithm_config["loss"] = ["notn2v"]

    with pytest.raises(ValueError):
        Algorithm(**algorithm_config)


def test_loss_value(minimum_config):
    """Test that we can instantiate a config with a single loss or a list of
    losses."""
    algorithm_config = minimum_config["algorithm"]

    # list
    algorithm_config["loss"] = ["n2v", "pn2v"]
    my_algo = Algorithm(**algorithm_config)
    assert my_algo.loss == algorithm_config["loss"]

    # single value throws error
    algorithm_config["loss"] = "n2v"

    with pytest.raises(ValidationError):
        Algorithm(**algorithm_config)


def test_wrong_model_value(minimum_config):
    """Test that we cannot instantiate a config with wrong loss value."""

    algorithm_config = minimum_config["algorithm"]
    algorithm_config["model"] = "wrongmodel"

    with pytest.raises(ValueError):
        Algorithm(**algorithm_config)


def test_wrong_manipulator_value(minimum_config):
    """Test that we cannot instantiate a config with wrong loss value."""

    algorithm_config = minimum_config["algorithm"]
    algorithm_config["pixel_manipulation"] = "notn2v"

    with pytest.raises(ValueError):
        Algorithm(**algorithm_config)


@pytest.mark.parametrize("mask_pixel_percentage", [-10, 0, 5.1])
def test_wrong_mask_pixel_percentage(minimum_config, mask_pixel_percentage):
    """Test that we cannot instantiate a config with wrong number of masked
    pixels."""

    algorithm_config = minimum_config["algorithm"]
    algorithm_config["mask_pixel_percentage"] = mask_pixel_percentage

    with pytest.raises(ValueError):
        Algorithm(**algorithm_config)


@pytest.mark.parametrize("conv_dims", [2, 3])
def test_conv_dims(minimum_config, conv_dims):
    """Test that we can instantiate a config with a valid conv_dims."""
    algorithm_config = minimum_config["algorithm"]
    algorithm_config["conv_dims"] = conv_dims

    # instantiate model
    my_algo = Algorithm(**algorithm_config)
    assert my_algo.conv_dims == conv_dims


@pytest.mark.parametrize("conv_dims", [-1, 0, 1, 4])
def test_wrong_conv_dims(minimum_config, conv_dims):
    """Test that we cannot instantiate a config with wrong conv_dims."""
    algorithm_config = minimum_config["algorithm"]
    algorithm_config["conv_dims"] = conv_dims

    # instantiate model
    with pytest.raises(ValueError):
        Algorithm(**algorithm_config)


@pytest.mark.parametrize("model_depth", [2, 3, 4, 5])
def test_model_depth(minimum_config, model_depth):
    """Test that we can instantiate a config with a valid model_depth."""
    algorithm_config = minimum_config["algorithm"]
    algorithm_config["depth"] = model_depth

    # instantiate model
    my_algo = Algorithm(**algorithm_config)
    assert my_algo.depth == model_depth


@pytest.mark.parametrize("model_depth", [-1, 0, 1, 6])
def test_wrong_model_depth(minimum_config, model_depth):
    """Test that we cannot instantiate a config with wrong model_depth."""
    algorithm_config = minimum_config["algorithm"]
    algorithm_config["depth"] = model_depth

    # instantiate model
    with pytest.raises(ValueError):
        Algorithm(**algorithm_config)
