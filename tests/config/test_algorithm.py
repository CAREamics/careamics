import pytest

from careamics.config.algorithm import Algorithm
from careamics.config.models import UNet
from careamics.config.noise_models import NoiseModel
from careamics.config.transforms import Transform


def test_algorithm_transforms():
    Transform(name="flip", parameters={"p": 0.5})


def test_algorithm_transforms_defaults():
    Transform(name="flip")


def test_algorithm_transforms_custom():
    Transform(
        name="DefaultManipulateN2V",
        parameters={"masked_pixel_percentage": 0.2, "roi_size": 11},
    )


def test_algorithm_transforms_custom_defaults():
    Transform(name="DefaultManipulateN2V")


def test_algorithm_transforms_wrong_name():
    with pytest.raises(ValueError):
        Transform(name="flip1", parameters={"p": 0.5})


def test_algorithm_transforms_wrong_parameters():
    with pytest.raises(ValueError):
        Transform(name="flip", parameters={"pa": 1})


def test_algorithm_noise_model():
    d = {
        "model_type": "hist",
        "parameters": {"min_value": 324, "max_value": 3465},
    }
    NoiseModel(**d)


@pytest.mark.parametrize("depth", [1, 5, 10])
def test_unet_parameters_depth(complete_config: dict, depth: int):
    """Test that UNet accepts depth between 1 and 10."""
    model_params = complete_config["algorithm"]["model"]["parameters"]
    model_params["depth"] = depth

    model = UNet(**model_params)
    assert model.depth == depth


@pytest.mark.parametrize("depth", [-1, 11])
def test_unet_parameters_wrong_depth(complete_config: dict, depth: int):
    """Test that wrong depth cause an error."""
    model_params = complete_config["algorithm"]["model"]["parameters"]
    model_params["depth"] = depth

    with pytest.raises(ValueError):
        UNet(**model_params)


@pytest.mark.parametrize("num_channels_init", [8, 16, 32, 96, 128])
def test_unet_parameters_num_channels_init(
    complete_config: dict, num_channels_init: int
):
    """Test that UNet accepts num_channels_init as a power of two and
    minimum 8."""
    model_params = complete_config["algorithm"]["model"]["parameters"]
    model_params["num_channels_init"] = num_channels_init

    model = UNet(**model_params)
    assert model.num_channels_init == num_channels_init


@pytest.mark.parametrize("num_channels_init", [2, 17, 127])
def test_unet_parameters_wrong_num_channels_init(
    complete_config: dict, num_channels_init: int
):
    """Test that wrong num_channels_init cause an error."""
    model_params = complete_config["algorithm"]["model"]["parameters"]
    model_params["num_channels_init"] = num_channels_init

    with pytest.raises(ValueError):
        UNet(**model_params)


@pytest.mark.parametrize("roi_size", [5, 9, 15])
def test_parameters_roi_size(complete_config: dict, roi_size: int):
    """Test that Algorithm accepts roi_size as an even number within the
    range [3, 21]."""
    complete_config["algorithm"]["masking_strategy"]["parameters"][
        "roi_size"
    ] = roi_size
    algorithm = Algorithm(**complete_config["algorithm"])
    assert algorithm.masking_strategy.parameters["roi_size"] == roi_size


@pytest.mark.parametrize("roi_size", [2, 4, 23])
def test_parameters_wrong_roi_size(complete_config: dict, roi_size: int):
    """Test that wrong num_channels_init cause an error."""
    complete_config["algorithm"]["masking_strategy"]["parameters"][
        "roi_size"
    ] = roi_size
    with pytest.raises(ValueError):
        Algorithm(**complete_config["algorithm"])


def test_unet_parameters_wrong_values_by_assigment(complete_config: dict):
    """Test that wrong values are not accepted through assignment."""
    model_params = complete_config["algorithm"]["model"]["parameters"]
    model = UNet(**model_params)

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
    algorithm["masking_strategy"]["parameters"][
        "masked_pixel_percentage"
    ] = masked_pixel_percentage

    algo = Algorithm(**algorithm)
    assert (
        algo.masking_strategy.parameters["masked_pixel_percentage"]
        == masked_pixel_percentage
    )


@pytest.mark.parametrize("masked_pixel_percentage", [0.01, 21])
def test_wrong_masked_pixel_percentage(
    complete_config: dict, masked_pixel_percentage: float
):
    """Test that Algorithm accepts the minimum configuration."""
    algorithm = complete_config["algorithm"]["masking_strategy"]["parameters"]
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

    algo.loss = algorithm["loss"]

    # model
    algo.model = algorithm["model"]
    with pytest.raises(ValueError):
        algo.model.architecture = "Unet"

    # is_3D
    algo.is_3D = algorithm["is_3D"]
    with pytest.raises(ValueError):
        algo.is_3D = 3

    # masking_strategy
    algo.masking_strategy = algorithm["masking_strategy"]
    with pytest.raises(ValueError):
        algo.masking_strategy.strategy_type = "mean"

    # masked_pixel_percentage
    # algo.masking_strategy = algorithm["masking_strategy"]
    # with pytest.raises(ValueError):
    #     algo.masking_strategy.parameters["masked_pixel_percentage"] = 0.01
    # TODO fix https://github.com/pydantic/pydantic/issues/7105

    # model_parameters
    algo.model.parameters = algorithm["model"]["parameters"]
    with pytest.raises(ValueError):
        algo.model.parameters = "params"


def test_algorithm_to_dict_minimum(minimum_config: dict):
    """ "Test that export to dict does not include optional values."""
    algorithm_minimum = Algorithm(**minimum_config["algorithm"]).model_dump()
    assert algorithm_minimum == minimum_config["algorithm"]

    assert "loss" in algorithm_minimum
    assert "model" in algorithm_minimum
    assert "is_3D" in algorithm_minimum
    # TODO revise set of defaults


def test_algorithm_to_dict_complete(complete_config: dict):
    """ "Test that export to dict does not include optional values."""
    algorithm_complete = Algorithm(**complete_config["algorithm"]).model_dump()
    assert algorithm_complete == complete_config["algorithm"]
    # TODO values are hardcoded in the fixture, is it ok ?
    assert "loss" in algorithm_complete
    assert "model" in algorithm_complete
    assert "is_3D" in algorithm_complete
    assert "masking_strategy" in algorithm_complete
