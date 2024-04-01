"""
Loss factory module.

This module contains a factory function for creating loss functions.
"""

from typing import Callable

from careamics.config import Configuration
from careamics.config.algorithm import Loss

from .losses import n2v_loss


def create_loss_function(config: Configuration) -> Callable:
    """
    Create loss function based on Configuration.

    Parameters
    ----------
    config : Configuration
        Configuration.

    Returns
    -------
    Callable
        Loss function.

    Raises
    ------
    NotImplementedError
        If the loss is unknown.
    """
    loss_type = config.algorithm.loss

    if loss_type == Loss.N2V:
        return n2v_loss
    else:
        raise NotImplementedError(f"Loss {loss_type} is not yet supported.")
