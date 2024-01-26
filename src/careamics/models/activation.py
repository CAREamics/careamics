from typing import Callable, Union

import torch.nn as nn

from ..config.architectures.architectures import Activation


def get_activation(activation: Union[Activation, str]) -> Callable:
    """
    Get activation function.

    Parameters
    ----------
    activation : str
        Activation function name.

    Returns
    -------
    Callable
        Activation function.
    """
    if activation == Activation.RELU:
        return nn.ReLU()
    elif activation == Activation.LEAKYRELU:
        return nn.LeakyReLU()
    elif activation == Activation.TANH:
        return nn.Tanh()
    elif activation == Activation.SIGMOID:
        return nn.Sigmoid()
    elif activation == Activation.SOFTMAX:
        return nn.Softmax(dim=1)
    elif activation == Activation.NONE:
        return nn.Identity()
    else:
        raise ValueError(f"Activation {activation} not supported.")

