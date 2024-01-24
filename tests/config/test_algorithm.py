import pytest

from careamics.config.torch_optim import (
    TorchLRScheduler,
    TorchOptimizer,
)
from careamics.config.algorithm import Algorithm, LrScheduler, Optimizer

# from careamics.config.noise_models import NoiseModel


# def test_algorithm_noise_model():
#     d = {
#         "model_type": "hist",
#         "parameters": {"min_value": 324, "max_value": 3465},
#     }
#     NoiseModel(**d)


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
    """Test that SGD optimizer fails if `lr` is not provided.
    
    Note: The SGD optimizer requires the `lr` parameter.
    """
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


def test_optimizer_to_dict_optional():
    """ "Test that export to dict includes optional values."""
    config = {
        "name": "Adam",
        "parameters": {
            "lr": 0.1,
            "betas": (0.1, 0.11),
        },
    }

    optim_minimum = Optimizer(**config).model_dump()
    assert optim_minimum == config


def test_optimizer_to_dict_default_optional():
    """ "Test that export to dict does not include default optional value."""
    config = {
        "name": "Adam",
        "parameters": {},
    }

    optim_minimum = Optimizer(**config).model_dump()
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


def test_scheduler_to_dict_optional():
    """ "Test that export to dict includes optional values."""
    scheduler_config = {
        "name": "ReduceLROnPlateau",
        "parameters": {
            "mode": "max",
            "factor": 0.3,
        },
    }

    scheduler_complete = LrScheduler(**scheduler_config).model_dump()
    assert scheduler_complete == scheduler_config


def test_scheduler_to_dict_default_optional():
    """ "Test that export to dict does not include optional value."""
    scheduler_config = {
        "name": "ReduceLROnPlateau",
        "parameters": {},
    }

    scheduler_complete = LrScheduler(**scheduler_config).model_dump()
    assert "parameters" not in scheduler_complete.keys()



def test_wrong_values_by_assigment(minimum_algorithm: dict):
    """Test that wrong values are not accepted through assignment."""
    algorithm = minimum_algorithm["algorithm"]
    algo = Algorithm(**algorithm)

    # loss
    algo.loss = algorithm["loss"]
    with pytest.raises(ValueError):
        algo.loss = "ms-meh"
    assert algo.loss == algorithm["loss"]

    # model
    algo.model = algorithm["model"]
    with pytest.raises(ValueError):
        algo.model.architecture = "Unet"

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

    