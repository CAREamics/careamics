"""
Convenience functions using torch.

These functions are used to control certain aspects and behaviours of PyTorch.
"""
import inspect
from typing import Dict, Tuple

import torch

from ..utils.logging import get_logger

logger = get_logger(__name__) # TODO are logger still needed?

# TODO remove the mandatory bc torch fails
def filter_parameters(
    func: type,
    user_params: dict,
) -> Tuple[dict, dict]:
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
    parameter_signature = inspect.signature(func).parameters

    # Get the list of all parameters
    possible_parameters = list(parameter_signature.keys())
    
    # Filter parameters to keep only those accepted by the function signature
    params_to_be_used = set(user_params.keys()) & set(possible_parameters)
    parameters = {key: user_params[key] for key in params_to_be_used}

    # Get the list of mandatory parameters
    mandatory_parameters = [
        str(parameter_signature[p])
        for p in parameter_signature
        if parameter_signature[p].default is inspect._empty
    ]

    # Find missing mandatory parameters
    missing_parameters = set(mandatory_parameters) - params_to_be_used

    return parameters, list(missing_parameters) 


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
