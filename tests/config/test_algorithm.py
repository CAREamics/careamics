import pytest

from careamics_restoration.config.algorithm import Algorithm, ModelParameters


@pytest.mark.parametrize("depth", [1, 5, 10])
def test_model_parameters_depth(complete_config: dict, depth: int):
    """Test that ModelParameters accepts depth between 1 and 10."""
    model_params = complete_config["algorithm"]["model_parameters"]
    model_params["depth"] = depth

    model = ModelParameters(**model_params)
    assert model.depth == depth


@pytest.mark.parametrize("depth", [-1, 11])
def test_model_parameters_wrong_depth(complete_config: dict, depth: int):
    """Test that wrong depth cause an error."""
    model_params = complete_config["algorithm"]["model_parameters"]
    model_params["depth"] = depth

    with pytest.raises(ValueError):
        ModelParameters(**model_params)


@pytest.mark.parametrize("num_channels_init", [8, 16, 32, 96, 128])
def test_model_parameters_num_channels_init(
    complete_config: dict, num_channels_init: int
):
    """Test that ModelParameters accepts num_channels_init as a power of two and
    minimum 8."""
    model_params = complete_config["algorithm"]["model_parameters"]
    model_params["num_channels_init"] = num_channels_init

    model = ModelParameters(**model_params)
    assert model.num_channels_init == num_channels_init


@pytest.mark.parametrize("num_channels_init", [2, 17, 127])
def test_model_parameters_wrong_num_channels_init(
    complete_config: dict, num_channels_init: int
):
    """Test that wrong num_channels_init cause an error."""
    model_params = complete_config["algorithm"]["model_parameters"]
    model_params["num_channels_init"] = num_channels_init

    with pytest.raises(ValueError):
        ModelParameters(**model_params)


def test_model_parameters_wrong_values_by_assigment(complete_config: dict):
    """Test that wrong values are not accepted through assignment."""
    model_params = complete_config["algorithm"]["model_parameters"]
    model = ModelParameters(**model_params)

    # depth
    model.depth = model_params["depth"]
    with pytest.raises(ValueError):
        model.depth = -1

    # number of channels
    model.num_channels_init = model_params["num_channels_init"]
    with pytest.raises(ValueError):
        model.num_channels_init = 2


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


def test_wrong_values_by_assigment(complete_config: dict):
    """Test that wrong values are not accepted through assignment."""
    algorithm = complete_config["algorithm"]
    algo = Algorithm(**algorithm)

    # loss
    algo.loss = algorithm["loss"]
    with pytest.raises(ValueError):
        algo.loss = "mse"

    # model
    algo.model = algorithm["model"]
    with pytest.raises(ValueError):
        algo.model = "unet"

    # is_3D
    algo.is_3D = algorithm["is_3D"]
    with pytest.raises(ValueError):
        algo.is_3D = 3

    # masking_strategy
    algo.masking_strategy = algorithm["masking_strategy"]
    with pytest.raises(ValueError):
        algo.masking_strategy = "mean"

    # masked_pixel_percentage
    algo.masked_pixel_percentage = algorithm["masked_pixel_percentage"]
    with pytest.raises(ValueError):
        algo.masked_pixel_percentage = 0.01

    # model_parameters
    algo.model_parameters = algorithm["model_parameters"]
    with pytest.raises(ValueError):
        algo.model_parameters = "params"


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
    # TODO values are hardcoded in the fixture, is it ok ?
    assert "loss" in algorithm_complete
    assert "model" in algorithm_complete
    assert "is_3D" in algorithm_complete
    assert "masking_strategy" in algorithm_complete
    assert "masked_pixel_percentage" in algorithm_complete
    assert "model_parameters" in algorithm_complete
    assert "depth" in algorithm_complete["model_parameters"]
    assert "num_channels_init" in algorithm_complete["model_parameters"]


def test_algorithm_to_dict_optionals(complete_config: dict):
    """ "Test that export to dict does not include optional values."""
    # change optional value to the default
    algo_config = complete_config["algorithm"]
    algo_config["model_parameters"] = {
        "depth": 2,
        "num_channels_init": 96,
    }
    algo_config["masking_strategy"] = "default"

    algorithm_complete = Algorithm(**complete_config["algorithm"]).model_dump()
    assert "model_parameters" not in algorithm_complete
    assert "masking_strategy" not in algorithm_complete
