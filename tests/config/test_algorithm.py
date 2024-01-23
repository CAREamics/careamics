import pytest

from careamics.config.torch_optim import (
    TorchLRScheduler,
    TorchOptimizer,
)
from careamics.config.algorithm import Algorithm, LrScheduler, Optimizer
from careamics.config.models import UNet
# from careamics.config.noise_models import NoiseModel


# def test_algorithm_noise_model():
#     d = {
#         "model_type": "hist",
#         "parameters": {"min_value": 324, "max_value": 3465},
#     }
#     NoiseModel(**d)


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

    # optimizer
    algo.optimizer = Optimizer(name=TorchOptimizer.Adam, parameters={"lr": 0.1})
    with pytest.raises(ValueError):
        algo.optimizer = "I'd rather not to."

    # lr_scheduler
    algo.lr_scheduler = LrScheduler(
        name=TorchLRScheduler.ReduceLROnPlateau, parameters={"factor": 0.1}
    )
    with pytest.raises(ValueError):
        algo.lr_scheduler = "Why don't you schedule it for once? :)"

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


@pytest.mark.parametrize(
    "optimizer_name, parameters",
    [
        (
            TorchOptimizer.Adam,
            {
                "lr": 0.08,
                "betas": (0.1, 0.11),
                "eps": 6e-08,
                "weight_decay": 0.2,
                "amsgrad": True,
            },
        ),
        (
            TorchOptimizer.SGD,
            {
                "lr": 0.11,
                "momentum": 5,
                "dampening": 1,
                "weight_decay": 8,
                "nesterov": True,
            },
        ),
    ],
)
def test_optimizer_parameters(optimizer_name: TorchOptimizer, parameters: dict):
    """Test optimizer parameters filtering.

    For parameters, see:
    https://pytorch.org/docs/stable/optim.html#algorithms
    """
    # add non valid parameter
    new_parameters = parameters.copy()
    new_parameters["some_random_one"] = 42

    # create optimizer and check that the parameters are filtered
    optimizer = Optimizer(name=optimizer_name, parameters=new_parameters)
    assert optimizer.parameters == parameters


def test_sgd_missing_parameter():
    """Test that SGD optimizer fails if `lr` is not provided"""
    with pytest.raises(ValueError):
        Optimizer(name=TorchOptimizer.SGD, parameters={})

    # test that it works if lr is provided
    optimizer = Optimizer(name=TorchOptimizer.SGD, parameters={"lr": 0.1})
    assert optimizer.parameters == {"lr": 0.1}


def test_optimizer_wrong_values_by_assignments():
    """Test that wrong values cause an error during assignment."""
    optimizer = Optimizer(name=TorchOptimizer.Adam, parameters={"lr": 0.08})

    # name
    optimizer.name = TorchOptimizer.SGD
    with pytest.raises(ValueError):
        optimizer.name = "MyOptim"

    # parameters
    optimizer.parameters = {"lr": 0.1}
    with pytest.raises(ValueError):
        optimizer.parameters = "lr = 0.3"


def test_optimizer_to_dict_minimum(minimum_config: dict):
    """ "Test that export to dict does not include optional value."""
    optim_minimum = Optimizer(**minimum_config["training"]["optimizer"]).model_dump()
    assert optim_minimum == minimum_config["training"]["optimizer"]

    assert "name" in optim_minimum.keys()
    assert "parameters" not in optim_minimum.keys()


def test_optimizer_to_dict_complete(complete_config: dict):
    """ "Test that export to dict does include optional value."""
    optim_minimum = Optimizer(**complete_config["training"]["optimizer"]).model_dump()
    assert optim_minimum == complete_config["training"]["optimizer"]

    assert "name" in optim_minimum.keys()
    assert "parameters" in optim_minimum.keys()


def test_optimizer_to_dict_optional(complete_config: dict):
    """ "Test that export to dict does not include optional value."""
    optim_config = complete_config["training"]["optimizer"]
    optim_config["parameters"] = {}

    optim_minimum = Optimizer(**optim_config).model_dump()
    assert "name" in optim_minimum.keys()
    assert "parameters" not in optim_minimum.keys()


@pytest.mark.parametrize(
    "lr_scheduler_name, parameters",
    [
        (
            TorchLRScheduler.ReduceLROnPlateau,
            {
                "mode": "max",
                "factor": 0.3,
                "patience": 5,
                "threshold": 0.003,
                "threshold_mode": "abs",
                "cooldown": 3,
                "min_lr": 0.1,
                "eps": 5e-08,
            },
        ),
        (
            TorchLRScheduler.StepLR,
            {
                "step_size": 2,
                "gamma": 0.3,
                "last_epoch": -5,
            },
        ),
    ],
)
def test_scheduler_parameters(lr_scheduler_name: TorchLRScheduler, parameters: dict):
    """Test lr scheduler parameters filtering.

    For parameters, see:
    https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    """
    # add non valid parameter
    new_parameters = parameters.copy()
    new_parameters["some_random_one"] = 42

    # create optimizer and check that the parameters are filtered
    lr_scheduler = LrScheduler(name=lr_scheduler_name, parameters=new_parameters)
    assert lr_scheduler.parameters == parameters


def test_scheduler_missing_parameter():
    """Test that StepLR scheduler fails if `step_size` is not provided"""
    with pytest.raises(ValueError):
        LrScheduler(name=TorchLRScheduler.StepLR, parameters={})

    # test that it works if lr is provided
    lr_scheduler = LrScheduler(
        name=TorchLRScheduler.StepLR, parameters={"step_size": "5"}
    )
    assert lr_scheduler.parameters == {"step_size": "5"}


def test_scheduler_wrong_values_by_assignments():
    """Test that wrong values cause an error during assignment."""
    scheduler = LrScheduler(
        name=TorchLRScheduler.ReduceLROnPlateau, parameters={"factor": 0.3}
    )

    # name
    scheduler.name = TorchLRScheduler.ReduceLROnPlateau
    with pytest.raises(ValueError):
        # this fails because the step parameter is missing
        scheduler.name = TorchLRScheduler.StepLR

    with pytest.raises(ValueError):
        scheduler.name = "Schedule it yourself!"

    # parameters
    scheduler.name = TorchLRScheduler.ReduceLROnPlateau
    scheduler.parameters = {"factor": 0.1}
    with pytest.raises(ValueError):
        scheduler.parameters = "factor = 0.3"


def test_scheduler_to_dict_minimum(minimum_config: dict):
    """ "Test that export to dict does not include optional value."""
    scheduler_minimum = LrScheduler(
        **minimum_config["training"]["lr_scheduler"]
    ).model_dump()
    assert scheduler_minimum == minimum_config["training"]["lr_scheduler"]

    assert "name" in scheduler_minimum.keys()
    assert "parameters" not in scheduler_minimum.keys()


def test_scheduler_to_dict_complete(complete_config: dict):
    """ "Test that export to dict does include optional value."""
    scheduler_complete = LrScheduler(
        **complete_config["training"]["lr_scheduler"]
    ).model_dump()
    assert scheduler_complete == complete_config["training"]["lr_scheduler"]

    assert "name" in scheduler_complete.keys()
    assert "parameters" in scheduler_complete.keys()


def test_scheduler_to_dict_optional(complete_config: dict):
    """ "Test that export to dict does not include optional value."""
    scheduler_config = complete_config["training"]["lr_scheduler"]
    scheduler_config["parameters"] = {}

    scheduler_complete = LrScheduler(**scheduler_config).model_dump()

    assert "name" in scheduler_complete.keys()
    assert "parameters" not in scheduler_complete.keys()
