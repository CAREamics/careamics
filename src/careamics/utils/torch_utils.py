"""
Convenience functions using torch.

These functions are used to control certain aspects and behaviours of PyTorch.
"""

import inspect
from typing import Dict, Union

import torch

from careamics.config.support import SupportedOptimizer, SupportedScheduler

from ..utils.logging import get_logger

logger = get_logger(__name__)  # TODO are logger still needed?


def filter_parameters(
    func: type,
    user_params: dict,
) -> dict:
    """
    Filter parameters according to the function signature.

    Parameters
    ----------
    func : type
        Class object.
    user_params : Dict
        User provided parameters.

    Returns
    -------
    Dict
        Parameters matching `func`'s signature.
    """
    # Get the list of all default parameters
    default_params = list(inspect.signature(func).parameters.keys())

    # Filter matching parameters
    params_to_be_used = set(user_params.keys()) & set(default_params)

    return {key: user_params[key] for key in params_to_be_used}


def get_optimizer(name: str) -> torch.optim.Optimizer:
    """
    Return the optimizer class given its name.

    Parameters
    ----------
    name : str
        Optimizer name.

    Returns
    -------
    torch.nn.Optimizer
        Optimizer class.
    """
    if name not in SupportedOptimizer:
        raise NotImplementedError(f"Optimizer {name} is not yet supported.")

    return getattr(torch.optim, name)


def get_optimizers() -> Dict[str, str]:
    """
    Return the list of all optimizers available in torch.optim.

    Returns
    -------
    Dict
        Optimizers available in torch.optim.
    """
    optims = {}
    for name, obj in inspect.getmembers(torch.optim):
        if inspect.isclass(obj) and issubclass(obj, torch.optim.Optimizer):
            if name != "Optimizer":
                optims[name] = name
    return optims


def get_scheduler(
    name: str,
) -> Union[
    torch.optim.lr_scheduler.LRScheduler,
    torch.optim.lr_scheduler.ReduceLROnPlateau,
]:
    """
    Return the scheduler class given its name.

    Parameters
    ----------
    name : str
        Scheduler name.

    Returns
    -------
    Union
        Scheduler class.
    """
    if name not in SupportedScheduler:
        raise NotImplementedError(f"Scheduler {name} is not yet supported.")

    return getattr(torch.optim.lr_scheduler, name)


def get_schedulers() -> Dict[str, str]:
    """
    Return the list of all schedulers available in torch.optim.lr_scheduler.

    Returns
    -------
    Dict
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
