import pytest

from careamics.config.support.supported_optimizers import (
    SupportedScheduler,
    SupportedOptimizer
)
from careamics.config.optimizers import OptimizerModel, LrSchedulerModel


@pytest.mark.parametrize(
    "optimizer_name, parameters",
    [
        (
            SupportedOptimizer.Adam.value,
            {
                "lr": 0.08,
                "betas": (0.1, 0.11),
                "eps": 6e-08,
                "weight_decay": 0.2,
                "amsgrad": True,
            },
        ),
        (
            SupportedOptimizer.SGD.value,
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
def test_optimizer_parameters(optimizer_name: SupportedOptimizer, parameters: dict):
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
        OptimizerModel(name=SupportedOptimizer.SGD.value, parameters={})

    # test that it works if lr is provided
    optimizer = OptimizerModel(
        name=SupportedOptimizer.SGD.value, 
        parameters={"lr": 0.1}
    )
    assert optimizer.parameters == {"lr": 0.1}


def test_optimizer_wrong_values_by_assignments():
    """Test that wrong values cause an error during assignment."""
    optimizer = OptimizerModel(name=SupportedOptimizer.Adam.value, parameters={"lr": 0.08})

    # name
    optimizer.name = SupportedOptimizer.SGD.value
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

    optim_minimum = OptimizerModel(**config).model_dump(exclude_defaults=True)
    assert "parameters" not in optim_minimum.keys()


@pytest.mark.parametrize(
    "lr_scheduler_name, parameters",
    [
        (
            SupportedScheduler.ReduceLROnPlateau.value,
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
            SupportedScheduler.StepLR.value,
            {
                "step_size": 2,
                "gamma": 0.3,
                "last_epoch": -5,
            },
        ),
    ],
)
def test_scheduler_parameters(lr_scheduler_name: SupportedScheduler, parameters: dict):
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
        LrSchedulerModel(name=SupportedScheduler.StepLR.value, parameters={})

    # test that it works if lr is provided
    lr_scheduler = LrSchedulerModel(
        name=SupportedScheduler.StepLR.value, parameters={"step_size": "5"}
    )
    assert lr_scheduler.parameters == {"step_size": "5"}


def test_scheduler_wrong_values_by_assignments():
    """Test that wrong values cause an error during assignment."""
    scheduler = LrSchedulerModel(
        name=SupportedScheduler.ReduceLROnPlateau.value, parameters={"factor": 0.3}
    )

    # name
    scheduler.name = SupportedScheduler.ReduceLROnPlateau.value
    with pytest.raises(ValueError):
        # this fails because the step parameter is missing
        scheduler.name = SupportedScheduler.StepLR.value

    with pytest.raises(ValueError):
        scheduler.name = "Schedule it yourself!"

    # parameters
    scheduler.name = SupportedScheduler.ReduceLROnPlateau.value
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

    scheduler_complete = LrSchedulerModel(**scheduler_config).model_dump(
        exclude_defaults=True
    )
    assert "parameters" not in scheduler_complete.keys()


