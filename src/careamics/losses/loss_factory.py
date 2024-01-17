"""
Loss factory module.

This module contains a factory function for creating loss functions.
"""
from typing import Callable, Type, Union

from ..config import Configuration
from ..config.algorithm import Loss
from ..config.noise_models import NoiseModelType
from .losses import dice_loss, mae_loss, mse_loss, n2v_loss, pn2v_loss
from .noise_models import GaussianMixtureNoiseModel, HistogramNoiseModel


# TODO add tests
def create_loss_function(loss_type: Loss) -> Callable:
    """Create loss function based on Configuration.

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
    if loss_type == Loss.N2V:
        return n2v_loss

    elif loss_type == Loss.PN2V:
        return pn2v_loss

    elif loss_type == Loss.MAE:
        return mae_loss

    elif loss_type == Loss.MSE:
        return mse_loss

    elif loss_type == Loss.DICE:
        return dice_loss

    else:
        raise NotImplementedError(f"Loss {loss_type} is not yet supported.")


def create_noise_model(
    config: Configuration,
) -> Type[Union[HistogramNoiseModel, GaussianMixtureNoiseModel, None]]:
    """Create loss model based on Configuration.

    Parameters
    ----------
    config : Configuration
        Configuration.

    Returns
    -------
    Noise model

    Raises
    ------
    NotImplementedError
        If the noise model is unknown.
    """
    noise_model_type = (
        config.algorithm.noise_model.model_type
        if config.algorithm.noise_model
        else None
    )

    if noise_model_type == NoiseModelType.HIST:
        return HistogramNoiseModel

    elif noise_model_type == NoiseModelType.GMM:
        return GaussianMixtureNoiseModel

    elif noise_model_type is None:
        return None

    else:
        raise NotImplementedError(
            f"Noise model {noise_model_type} is not yet supported."
        )
