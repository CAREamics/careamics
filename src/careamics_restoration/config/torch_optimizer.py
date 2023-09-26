import inspect
from enum import Enum
from typing import Dict

from torch import optim


class TorchOptimizer(str, Enum):
    """List of all optimizers available in torch.optim.

    Currently only supports Adam and SGD.
    """

    # ASGD = "ASGD"
    # Adadelta = "Adadelta"
    # Adagrad = "Adagrad"
    Adam = "Adam"
    # AdamW = "AdamW"
    # Adamax = "Adamax"
    # LBFGS = "LBFGS"
    # NAdam = "NAdam"
    # RAdam = "RAdam"
    # RMSprop = "RMSprop"
    # Rprop = "Rprop"
    SGD = "SGD"
    # SparseAdam = "SparseAdam"


# TODO: Test which schedulers are compatible and if not, how to make them compatible
# (if we want to support them)
class TorchLRScheduler(str, Enum):
    """List of all schedulers available in torch.optim.lr_scheduler.

    Currently only supports ReduceLROnPlateau and StepLR.
    """

    # ChainedScheduler = "ChainedScheduler"
    # ConstantLR = "ConstantLR"
    # CosineAnnealingLR = "CosineAnnealingLR"
    # CosineAnnealingWarmRestarts = "CosineAnnealingWarmRestarts"
    # CyclicLR = "CyclicLR"
    # ExponentialLR = "ExponentialLR"
    # LambdaLR = "LambdaLR"
    # LinearLR = "LinearLR"
    # MultiStepLR = "MultiStepLR"
    # MultiplicativeLR = "MultiplicativeLR"
    # OneCycleLR = "OneCycleLR"
    # PolynomialLR = "PolynomialLR"
    ReduceLROnPlateau = "ReduceLROnPlateau"
    # SequentialLR = "SequentialLR"
    StepLR = "StepLR"


def get_parameters(
    func: type,
    user_params: dict,
) -> dict:
    """Filter parameters according to `func`'s signature.

    Parameters
    ----------
    func : type
        Class object
    user_params : Dict
        User provided parameters

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
    """Returns the list of all optimizers available in torch.optim."""
    optims = {}
    for name, obj in inspect.getmembers(optim):
        if inspect.isclass(obj) and issubclass(obj, optim.Optimizer):
            if name != "Optimizer":
                optims[name] = name
    return optims


def get_schedulers() -> Dict[str, str]:
    """Returns the list of all schedulers available in torch.optim.lr_scheduler."""
    schedulers = {}
    for name, obj in inspect.getmembers(optim.lr_scheduler):
        if inspect.isclass(obj) and issubclass(obj, optim.lr_scheduler.LRScheduler):
            if "LRScheduler" not in name:
                schedulers[name] = name
        elif name == "ReduceLROnPlateau":  # somewhat not a subclass of LRScheduler
            schedulers[name] = name
    return schedulers
