import pytest

from careamics_restoration.config.torch_optimizer import (
    TorchLRScheduler,
    TorchOptimizer,
)
from careamics_restoration.config.training import AMP, LrScheduler, Optimizer, Training


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
def test_lr_scheduler_parameters(lr_scheduler_name: TorchLRScheduler, parameters: dict):
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


@pytest.mark.parametrize("init_scale", [512, 1024, 65536])
def test_amp_init_scale(init_scale: int):
    """Test AMP init_scale parameter."""
    amp = AMP(use=True, init_scale=init_scale)
    assert amp.init_scale == init_scale


@pytest.mark.parametrize("init_scale", [511, 1088, 65537])
def test_amp_wrong_init_scale(init_scale: int):
    """Test wrong AMP init_scale parameter."""
    with pytest.raises(ValueError):
        AMP(use=True, init_scale=init_scale)


@pytest.mark.parametrize("num_epochs", [1, 2, 4, 9000])
def test_training_num_epochs(minimum_config: dict, num_epochs: int):
    """Test that Training accepts num_epochs greater than 0."""
    training = minimum_config["training"]
    training["num_epochs"] = num_epochs

    training = Training(**training)
    assert training.num_epochs == num_epochs


@pytest.mark.parametrize("num_epochs", [-1, 0])
def test_training_wrong_num_epochs(minimum_config: dict, num_epochs: int):
    """Test that wrong number of epochs cause an error."""
    training = minimum_config["training"]
    training["num_epochs"] = num_epochs

    Training(**training)
    print(training)
