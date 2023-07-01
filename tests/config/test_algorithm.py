import pytest

from careamics_restoration.config.algorithm import Algorithm, ModelParameters


@pytest.mark.parametrize("depth", [1, 5, 10])
def test_model_parameters_depth(complete_config: dict, depth: int):
    """Test that ModelParameters accepts depth between 1 and 10."""
    model_params = complete_config["algorithm"]["model_parameters"]
    model_params["depth"] = depth

    model_params = ModelParameters(**model_params)
    assert model_params.depth == depth


@pytest.mark.parametrize("depth", [-1, 11])
def test_model_parameters_wrong_depth(complete_config: dict, depth: int):
    """Test that wrong depth cause an error."""
    model_params = complete_config["algorithm"]["model_parameters"]
    model_params["depth"] = depth

    with pytest.raises(ValueError):
        ModelParameters(**model_params)


@pytest.mark.parametrize("num_filters_base", [8, 16, 32, 96, 128])
def test_model_parameters_num_filters_base(
    complete_config: dict, num_filters_base: int
):
    """Test that ModelParameters accepts num_filters_base as a power of two and
    minimum 8."""
    model_params = complete_config["algorithm"]["model_parameters"]
    model_params["num_filters_base"] = num_filters_base

    model_params = ModelParameters(**model_params)
    assert model_params.num_filters_base == num_filters_base


@pytest.mark.parametrize("num_filters_base", [2, 17, 127])
def test_model_parameters_wrong_num_filters_base(
    complete_config: dict, num_filters_base: int
):
    """Test that wrong num_filters_base cause an error."""
    model_params = complete_config["algorithm"]["model_parameters"]
    model_params["num_filters_base"] = num_filters_base

    with pytest.raises(ValueError):
        ModelParameters(**model_params)


@pytest.mark.parametrize("masked_pixel_percentage", [0.1, 0.2, 5, 20])
def test_masked_pixel_percentage(complete_config: dict, masked_pixel_percentage: float):
    """Test that Algorithm accepts the minimum configuration."""
    algorithm = complete_config["algorithm"]
    algorithm["masked_pixel_percentage"] = masked_pixel_percentage

    algo = Algorithm(**algorithm)
    assert algo.masked_pixel_percentage == masked_pixel_percentage


@pytest.mark.parametrize("masked_pixel_percentage", [0.01, 21])
def test_wrong_masked_pixel_percentage(
    complete_config: dict, masked_pixel_percentage: float
):
    """Test that Algorithm accepts the minimum configuration."""
    algorithm = complete_config["algorithm"]
    algorithm["masked_pixel_percentage"] = masked_pixel_percentage

    with pytest.raises(ValueError):
        Algorithm(**algorithm)


def test_algorithm_to_dict_minimum(minimum_config: dict):
    """ "Test that export to dict does not include optional values."""
    algorithm_minimum = Algorithm(**minimum_config["algorithm"]).model_dump()
    assert algorithm_minimum == minimum_config["algorithm"]

    assert "loss" in algorithm_minimum
    assert "model" in algorithm_minimum
    assert "is_3D" in algorithm_minimum
    assert "masking_strategy" not in algorithm_minimum
    assert "masked_pixel_percentage" not in algorithm_minimum
    assert "model_parameters" not in algorithm_minimum


def test_algorithm_to_dict_complete(complete_config: dict):
    """ "Test that export to dict does not include optional values."""
    algorithm_complete = Algorithm(**complete_config["algorithm"]).model_dump()
    assert algorithm_complete == complete_config["algorithm"]

    assert "loss" in algorithm_complete
    assert "model" in algorithm_complete
    assert "is_3D" in algorithm_complete
    assert "masking_strategy" in algorithm_complete
    assert "masked_pixel_percentage" in algorithm_complete
    assert "model_parameters" in algorithm_complete
    assert "depth" in algorithm_complete["model_parameters"]
    assert "num_filters_base" in algorithm_complete["model_parameters"]


def test_algorithm_to_dict_optionals(complete_config: dict):
    """ "Test that export to dict does not include optional values."""
    # change optional value to the default
    algo_config = complete_config["algorithm"]
    algo_config["model_parameters"] = {
        "depth": 2,
        "num_filters_base": 96,
    }
    algo_config["masking_strategy"] = "default"

    algorithm_complete = Algorithm(**complete_config["algorithm"]).model_dump()
    assert "model_parameters" not in algorithm_complete
    assert "masking_strategy" not in algorithm_complete
