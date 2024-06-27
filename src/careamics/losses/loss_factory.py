"""
Loss factory module.

This module contains a factory function for creating loss functions.
"""

from torch import Tensor as tensor
from typing import Callable, Union
from dataclasses import dataclass
from ..config.support import SupportedLoss
from .fcn.losses import mae_loss, mse_loss, n2v_loss
from .lvae.losses import denoisplit_loss, musplit_loss


#TODO Add similar dataclass fro Unet ?
@dataclass
class LVAELossParameters:
    prediction: tensor
    prediction_data: tensor
    targets: tensor
    inputs: tensor
    mask: tensor
    current_epoch: int
    reconstruction_weight: float
    denoisplit_weight: float
    usplit_weight: float
    kl_annealing: bool
    kl_start: int
    kl_annealtime: int
    kl_weight: float
    non_stochastic: bool



def loss_factory(loss: Union[SupportedLoss: str]) -> Callable:
    """Return loss function.

    Parameters
    ----------
    loss : Union[SupportedLoss: str]
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

    else:
        raise NotImplementedError(f"Loss {loss} is not yet supported.")
