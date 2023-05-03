import sys
import inspect
from enum import Enum

from torch import optim


# extract Optimizer child classes from torch.optim
optims = {}
for name, obj in inspect.getmembers(optim):
    if inspect.isclass(obj):
        if name != "Optimizer":
            optims[name] = name


TorchOptimizer = Enum("TorchOptimizer", optims)


print([e.value for e in TorchOptimizer])
