"""
Convenience functions using torch.

These functions are used to control certain aspects and behaviours of PyTorch.
"""

import inspect
import platform

import torch

from careamics.config.support import SupportedOptimizer, SupportedScheduler

# TODO move these functions elsewhere


# TODO: this should be used, but it isn't anymore?
def get_device() -> str:
    """
    Get the device on which operations take place.

    Returns
    -------
    str
        The device on which operations take place, e.g. "cuda", "cpu" or "mps".
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available() and platform.processor() in (
        "arm",
        "arm64",
    ):
        device = "mps"
    else:
        device = "cpu"

    return device


# TODO misplaced?
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
    user_params : dict
        User provided parameters.

    Returns
    -------
    dict
        Parameters matching `func`'s signature.
    """
    # Get the list of all default parameters
    default_params = list(inspect.signature(func).parameters.keys())

    # Filter matching parameters
    params_to_be_used = set(user_params.keys()) & set(default_params)

    return {key: user_params[key] for key in params_to_be_used}


def get_optimizer(name: str) -> type[torch.optim.Optimizer]:
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
    try:
        SupportedOptimizer(name)
    except ValueError as e:
        raise NotImplementedError(f"Optimizer {name} is not yet supported.") from e

    return getattr(torch.optim, name)


# TODO unused but in tests
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


def get_scheduler(
    name: str,
) -> type[torch.optim.lr_scheduler.ReduceLROnPlateau]:
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
    try:
        SupportedScheduler(name)
    except ValueError as e:
        raise NotImplementedError(f"Scheduler {name} is not yet supported.") from e

    return getattr(torch.optim.lr_scheduler, name)


# TODO unused but in tests
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
