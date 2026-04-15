"""
Loss factory module.

This module contains a factory function for creating loss functions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Union

from careamics.config.support import SupportedLoss

from .lvae_losses import (
    denoisplit_loss,
    denoisplit_musplit_loss,
    musplit_loss,
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
        return musplit_loss

    elif loss == SupportedLoss.DENOISPLIT:
        return denoisplit_loss

    elif loss == SupportedLoss.DENOISPLIT_MUSPLIT:
        return denoisplit_musplit_loss

    else:
        raise NotImplementedError(f"Loss {loss} is not supported for LVAE models.")
