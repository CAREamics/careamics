import inspect

import torch


def get_optimizers() -> dict[str, str]:
    """
    Return the list of all optimizers available in torch.optim.

    Returns
    -------
    dict
        Optimizers available in torch.optim.
    """
    optims = {}
    for name, obj in inspect.getmembers(torch.optim):
        if inspect.isclass(obj) and issubclass(obj, torch.optim.Optimizer):
            if name != "Optimizer":
                optims[name] = name
    return optims


def get_schedulers() -> dict[str, str]:
    """
    Return the list of all schedulers available in torch.optim.lr_scheduler.

    Returns
    -------
    dict
        Schedulers available in torch.optim.lr_scheduler.
    """
    schedulers = {}
    for name, obj in inspect.getmembers(torch.optim.lr_scheduler):
        if inspect.isclass(obj) and issubclass(
            obj, torch.optim.lr_scheduler.LRScheduler
        ):
            if "LRScheduler" not in name:
                schedulers[name] = name
        elif name == "ReduceLROnPlateau":  # somewhat not a subclass of LRScheduler
            schedulers[name] = name
    return schedulers


def test_get_schedulers_exist():
    """Test that the function `get_schedulers` return
    existing torch schedulers.
    """
    for scheduler in get_schedulers():
        assert hasattr(torch.optim.lr_scheduler, scheduler)


def test_get_optimizers_exist():
    """Test that the function `get_optimizers` return
    existing torch optimizers.
    """
    for optimizer in get_optimizers():
        assert hasattr(torch.optim, optimizer)
