import pytest
from pydantic.error_wrappers import ValidationError

from careamics_restoration.config.algorithm import Algorithm, ModelParameters


@pytest.mark.parametrize("depth", [1, 5, 10])
def test_model_parameters_depth(complete_config, depth):
    """Test that ModelParameters accepts depth between 1 and 10."""
    model_params = complete_config["algorithm"]["model_parameters"]
    model_params["depth"] = depth

    ModelParameters(**model_params)


@pytest.mark.parametrize("depth", [-1, 11])
def test_model_parameters_wrong_depth(complete_config, depth):
    """Test that ModelParameters accepts depth between 1 and 10."""
    model_params = complete_config["algorithm"]["model_parameters"]
    model_params["depth"] = depth

    with pytest.raises(ValidationError):
        ModelParameters(**model_params)


@pytest.mark.parametrize("num_filters_base", [8, 16, 32, 96, 128])
def test_model_parameters_num_filters_base(complete_config, num_filters_base):
    """Test that ModelParameters accepts num_filters_base as a power of two and
    minimum 8."""
    model_params = complete_config["algorithm"]["model_parameters"]
    model_params["num_filters_base"] = num_filters_base

    ModelParameters(**model_params)


@pytest.mark.parametrize("num_filters_base", [2, 17, 127])
def test_model_parameters_num_filters_base(complete_config, num_filters_base):
    """Test that wrong num_filters_base."""
    model_params = complete_config["algorithm"]["model_parameters"]
    model_params["num_filters_base"] = num_filters_base

    with pytest.raises(ValidationError):
        ModelParameters(**model_params)


@pytest.mark.parametrize("masked_pixel_percentage", [0.1, 0.2, 5, 20])
def test_masked_pixel_percentage(complete_config, masked_pixel_percentage):
    """Test that Algorithm accepts the minimum configuration."""
    algorithm = complete_config["algorithm"]
    algorithm["masked_pixel_percentage"] = masked_pixel_percentage

    Algorithm(**algorithm)


@pytest.mark.parametrize("masked_pixel_percentage", [0.01, 21])
def test_wrong_masked_pixel_percentage(complete_config, masked_pixel_percentage):
    """Test that Algorithm accepts the minimum configuration."""
    algorithm = complete_config["algorithm"]
    algorithm["masked_pixel_percentage"] = masked_pixel_percentage

    with pytest.raises(ValidationError):
        Algorithm(**algorithm)
