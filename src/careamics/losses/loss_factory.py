"""
Loss factory module.

This module contains a factory function for creating loss functions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Union

from torch import Tensor as tensor

from ..config.support import SupportedLoss
from .fcn.losses import mae_loss, mse_loss, n2v_loss
from .lvae.losses import denoisplit_loss, denoisplit_musplit_loss, musplit_loss


@dataclass
class FCNLossParameters:
    """Dataclass for FCN loss."""

    # TODO check
    prediction: tensor
    targets: tensor
    mask: tensor
    current_epoch: int
    loss_weight: float


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

    # elif loss_type == SupportedLoss.PN2V:
    #     return pn2v_loss

    elif loss == SupportedLoss.MAE:
        return mae_loss

    elif loss == SupportedLoss.MSE:
        return mse_loss

    elif loss == SupportedLoss.MUSPLIT:
        return musplit_loss

    elif loss == SupportedLoss.DENOISPLIT:
        return denoisplit_loss

    elif loss == SupportedLoss.DENOISPLIT_MUSPLIT:
        return denoisplit_musplit_loss

    else:
        raise NotImplementedError(f"Loss {loss} is not yet supported.")
