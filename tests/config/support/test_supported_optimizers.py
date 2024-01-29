from torch import optim

from careamics.config.support.supported_optimizers import (
    SupportedScheduler,
    SupportedOptimizer
)


def test_schedulers_exist():
    """Test that `SupportedScheduler` contains existing torch schedulers."""
    for scheduler in SupportedScheduler:
        assert hasattr(optim.lr_scheduler, scheduler)


def test_optimizers_exist():
    """Test that `SupportedOptimizer` contains existing torch optimizers."""
    for optimizer in SupportedOptimizer:
        assert hasattr(optim, optimizer)
