"""
Loss factory module.

This module contains a factory function for creating loss functions.
"""

from typing import Callable, Union

from ..config.support import SupportedLoss
from .losses import mae_loss, mse_loss, n2v_loss


# TODO add tests
# TODO add custom?
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

    # elif loss_type == SupportedLoss.DICE:
    #     return dice_loss

    else:
        raise NotImplementedError(f"Loss {loss} is not yet supported.")
