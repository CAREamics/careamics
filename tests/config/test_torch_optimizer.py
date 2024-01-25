import pytest
from torch import optim

from careamics.config.torch_optim import (
    OptimizerModel,
    LrSchedulerModel,
    TorchLRSchedulers,
    TorchOptimizers,
    get_optimizers,
    get_schedulers,
)


def test_get_schedulers_exist():
    """Test that the function `get_schedulers` return
    existing torch schedulers.
    """
    for scheduler in get_schedulers():
        assert hasattr(optim.lr_scheduler, scheduler)


def test_torch_schedulers_exist():
    """Test that the enum `TorchLRScheduler` contains
    existing torch schedulers."""
    for scheduler in TorchLRSchedulers:
        assert hasattr(optim.lr_scheduler, scheduler)


def test_get_optimizers_exist():
    """Test that the function `get_optimizers` return
    existing torch optimizers.
    """
    for optimizer in get_optimizers():
        assert hasattr(optim, optimizer)


def test_optimizers_exist():
    """Test that the enum `TorchOptimizer` contains
    existing torch optimizers."""
    for optimizer in TorchOptimizers:
        assert hasattr(optim, optimizer)


@pytest.mark.parametrize(
    "optimizer_name, parameters",
    [
        (
            TorchOptimizers.Adam,
            {
                "lr": 0.08,
                "betas": (0.1, 0.11),
                "eps": 6e-08,
                "weight_decay": 0.2,
                "amsgrad": True,
            },
        ),
        (
            TorchOptimizers.SGD,
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
def test_optimizer_parameters(optimizer_name: TorchOptimizers, parameters: dict):
    """Test optimizer parameters filtering.

    For parameters, see:
    https://pytorch.org/docs/stable/optim.html#algorithms
    """
    # add non valid parameter
    new_parameters = parameters.copy()
    new_parameters["some_random_one"] = 42

    # create optimizer and check that the parameters are filtered
    optimizer = OptimizerModel(name=optimizer_name, parameters=new_parameters)
    assert optimizer.parameters == parameters


def test_sgd_missing_parameter():
    """Test that SGD optimizer fails if `lr` is not provided.
    
    Note: The SGD optimizer requires the `lr` parameter.
    """
    with pytest.raises(ValueError):
        OptimizerModel(name=TorchOptimizers.SGD, parameters={})

    # test that it works if lr is provided
    optimizer = OptimizerModel(name=TorchOptimizers.SGD, parameters={"lr": 0.1})
    assert optimizer.parameters == {"lr": 0.1}


def test_optimizer_wrong_values_by_assignments():
    """Test that wrong values cause an error during assignment."""
    optimizer = OptimizerModel(name=TorchOptimizers.Adam, parameters={"lr": 0.08})

    # name
    optimizer.name = TorchOptimizers.SGD
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

    optim_minimum = OptimizerModel(**config).model_dump()
    assert optim_minimum == config


def test_optimizer_to_dict_default_optional():
    """ "Test that export to dict does not include default optional value."""
    config = {
        "name": "Adam",
        "parameters": {},
    }

    optim_minimum = OptimizerModel(**config).model_dump()
    assert "parameters" not in optim_minimum.keys()


@pytest.mark.parametrize(
    "lr_scheduler_name, parameters",
    [
        (
            TorchLRSchedulers.ReduceLROnPlateau,
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
            TorchLRSchedulers.StepLR,
            {
                "step_size": 2,
                "gamma": 0.3,
                "last_epoch": -5,
            },
        ),
    ],
)
def test_scheduler_parameters(lr_scheduler_name: TorchLRSchedulers, parameters: dict):
    """Test lr scheduler parameters filtering.

    For parameters, see:
    https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    """
    # add non valid parameter
    new_parameters = parameters.copy()
    new_parameters["some_random_one"] = 42

    # create optimizer and check that the parameters are filtered
    lr_scheduler = LrSchedulerModel(name=lr_scheduler_name, parameters=new_parameters)
    assert lr_scheduler.parameters == parameters


def test_scheduler_missing_parameter():
    """Test that StepLR scheduler fails if `step_size` is not provided"""
    with pytest.raises(ValueError):
        LrSchedulerModel(name=TorchLRSchedulers.StepLR, parameters={})

    # test that it works if lr is provided
    lr_scheduler = LrSchedulerModel(
        name=TorchLRSchedulers.StepLR, parameters={"step_size": "5"}
    )
    assert lr_scheduler.parameters == {"step_size": "5"}


def test_scheduler_wrong_values_by_assignments():
    """Test that wrong values cause an error during assignment."""
    scheduler = LrSchedulerModel(
        name=TorchLRSchedulers.ReduceLROnPlateau, parameters={"factor": 0.3}
    )

    # name
    scheduler.name = TorchLRSchedulers.ReduceLROnPlateau
    with pytest.raises(ValueError):
        # this fails because the step parameter is missing
        scheduler.name = TorchLRSchedulers.StepLR

    with pytest.raises(ValueError):
        scheduler.name = "Schedule it yourself!"

    # parameters
    scheduler.name = TorchLRSchedulers.ReduceLROnPlateau
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

    scheduler_complete = LrSchedulerModel(**scheduler_config).model_dump()
    assert scheduler_complete == scheduler_config


def test_scheduler_to_dict_default_optional():
    """ "Test that export to dict does not include optional value."""
    scheduler_config = {
        "name": "ReduceLROnPlateau",
        "parameters": {},
    }

    scheduler_complete = LrSchedulerModel(**scheduler_config).model_dump()
    assert "parameters" not in scheduler_complete.keys()


