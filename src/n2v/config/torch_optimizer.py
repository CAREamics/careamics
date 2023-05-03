import inspect
from enum import Enum

from torch import optim


def get_optimizers():
    """Returns the list of all optimizers available in torch.optim"""
    optims = {}
    for name, obj in inspect.getmembers(optim):
        if inspect.isclass(obj):
            if name != "Optimizer":
                optims[name] = name
    return optims


TorchOptimizer = Enum("TorchOptimizer", get_optimizers())
