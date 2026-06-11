"""
Loss factory module.

This module contains a factory function for creating loss functions.
"""

from collections.abc import Callable
from typing import Union

from careamics.config.support import SupportedLoss

from .lvae_losses import (
    hdn_loss,
    microsplit_loss,
)


def lvae_loss_factory(loss: Union[SupportedLoss, str]) -> Callable:
    """Return loss function.

    Parameters
    ----------
    loss : Union[SupportedLoss, str]
        Requested loss.

    Returns
    -------
    Callable
        Loss function.

    Raises
    ------
    NotImplementedError
        If the loss is unknown.
    """
    if loss == SupportedLoss.MUSPLIT:
        return microsplit_loss

    elif loss == SupportedLoss.HDN:
        return hdn_loss

    else:
        raise NotImplementedError(f"Loss {loss} is not supported for LVAE models.")
