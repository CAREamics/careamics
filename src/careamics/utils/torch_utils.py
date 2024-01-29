"""
Convenience functions using torch.

These functions are used to control certain aspects and behaviours of PyTorch.
"""
import inspect
from typing import Dict

from torch import optim

from ..utils.logging import get_logger

logger = get_logger(__name__)


def get_parameters(
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


def get_optimizers() -> Dict[str, str]:
    """
    Return the list of all optimizers available in torch.optim.

    Returns
    -------
    Dict
        Optimizers available in torch.optim.
    """
    optims = {}
    for name, obj in inspect.getmembers(optim):
        if inspect.isclass(obj) and issubclass(obj, optim.Optimizer):
            if name != "Optimizer":
                optims[name] = name
    return optims


def get_schedulers() -> Dict[str, str]:
    """
    Return the list of all schedulers available in torch.optim.lr_scheduler.

    Returns
    -------
    Dict
        Schedulers available in torch.optim.lr_scheduler.
    """
    schedulers = {}
    for name, obj in inspect.getmembers(optim.lr_scheduler):
        if inspect.isclass(obj) and issubclass(obj, optim.lr_scheduler.LRScheduler):
            if "LRScheduler" not in name:
                schedulers[name] = name
        elif name == "ReduceLROnPlateau":  # somewhat not a subclass of LRScheduler
            schedulers[name] = name
    return schedulers
