from torch import optim

from careamics_restoration.config.torch_optimizer import (
    TorchLRScheduler,
    TorchOptimizer,
    get_optimizers,
    get_schedulers,
)

# TODO test get parameters


def test_schedulers_are_all_present():
    assert len(TorchLRScheduler) == len(get_schedulers())


def test_schedulers_exist():
    for scheduler in TorchLRScheduler:
        assert hasattr(optim.lr_scheduler, scheduler)


def test_optimizers_are_all_present():
    assert len(TorchOptimizer) == len(get_optimizers())


def test_optimizers_exist():
    for optimizer in TorchOptimizer:
        assert hasattr(optim, optimizer)
