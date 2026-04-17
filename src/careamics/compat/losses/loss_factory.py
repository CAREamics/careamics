"""
Loss factory module.

This module contains a factory function for creating loss functions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Union

from torch.nn import L1Loss, MSELoss

from careamics.config.support import SupportedLoss
from careamics.losses.n2v_losses import n2v_loss, pn2v_loss


def loss_factory(loss: Union[SupportedLoss, str]) -> Callable:
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
    if loss == SupportedLoss.N2V:
        return n2v_loss

    elif loss == SupportedLoss.PN2V:
        return pn2v_loss

    elif loss == SupportedLoss.MAE:
        return L1Loss()

    elif loss == SupportedLoss.MSE:
        return MSELoss()

    else:
        raise NotImplementedError(f"Loss {loss} is not supported.")
