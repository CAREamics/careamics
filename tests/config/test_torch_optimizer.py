from torch import optim

from careamics.config.torch_optimizer import (
    TorchLRScheduler,
    TorchOptimizer,
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
    for scheduler in TorchLRScheduler:
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
    for optimizer in TorchOptimizer:
        assert hasattr(optim, optimizer)
