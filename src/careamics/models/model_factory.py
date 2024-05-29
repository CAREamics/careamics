"""
Model factory.

Model creation factory functions.
"""

from typing import Union

import torch

from ..config.architectures import CustomModel, UNetModel, VAEModel, get_custom_model
from ..config.support import SupportedArchitecture
from ..utils import get_logger
from .unet import UNet

logger = get_logger(__name__)


def model_factory(
    model_configuration: Union[UNetModel, VAEModel, CustomModel]
) -> torch.nn.Module:
    """
    Deep learning model factory.

    Supported models are defined in careamics.config.SupportedArchitecture.

    Parameters
    ----------
    model_configuration : Union[UNetModel, VAEModel]
        Model configuration.

    Returns
    -------
    torch.nn.Module
        Model class.

    Raises
    ------
    NotImplementedError
        If the requested architecture is not implemented.
    """
    if model_configuration.architecture == SupportedArchitecture.UNET:
        return UNet(**model_configuration.model_dump())
    elif model_configuration.architecture == SupportedArchitecture.CUSTOM:
        assert isinstance(model_configuration, CustomModel)
        model = get_custom_model(model_configuration.name)

        return model(**model_configuration.model_dump())
    else:
        raise NotImplementedError(
            f"Model {model_configuration.architecture} is not implemented or unknown."
        )
