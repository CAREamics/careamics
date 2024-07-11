"""
Loss factory module.

This module contains a factory function for creating loss functions.
"""

from dataclasses import dataclass
from typing import Callable, Union

from torch import Tensor as tensor

from ..config.support import SupportedLoss
from .fcn.losses import mae_loss, mse_loss, n2v_loss
from .lvae.losses import denoisplit_loss, musplit_loss


@dataclass
class FCNLossParameters:
    """Dataclass for FCN loss."""

    # TODO check
    prediction: tensor
    targets: tensor
    mask: tensor
    current_epoch: int
    loss_weight: float


@dataclass
class LVAELossParameters:
    """Dataclass for LVAE loss."""

    prediction: tensor
    prediction_data: tensor # td_data
    targets: tensor
    inputs: tensor
    mask: tensor
    likelihood: Callable
    noise_model: Callable
    current_epoch: int
    reconstruction_weight: float = 1.0
    musplit_weight: float = 0.0
    denoisplit_weight: float = 1.0
    kl_annealing: bool = False
    kl_start: int = -1
    kl_annealtime: int = 10
    kl_weight: float = 1.0
    non_stochastic: bool = False


def loss_parameters_factory(
    type: SupportedLoss,
) -> Union[FCNLossParameters, LVAELossParameters]:
    """Return loss parameters.

    Parameters
    ----------
    type : SupportedLoss
        Requested loss.

    Returns
    -------
    Union[FCNLossParameters, LVAELossParameters]
        Loss parameters.

    Raises
    ------
    NotImplementedError
        If the loss is unknown.
    """
    if type in [SupportedLoss.N2V, SupportedLoss.MSE, SupportedLoss.MAE]:
        return FCNLossParameters

    elif type in [SupportedLoss.MUSPLIT, SupportedLoss.DENOISPLIT]:
        return LVAELossParameters

    else:
        raise NotImplementedError(f"Loss {type} is not yet supported.")


def loss_factory(loss: Union[SupportedLoss, str]) -> Callable:
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
