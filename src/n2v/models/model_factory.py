import logging

import torch

from ..config import Configuration
from ..utils import set_logging
from .models import UNET

logger = logging.getLogger(__name__)
set_logging(logger)


def create_model(config: Configuration) -> torch.nn.Module:
    """Builds a model based on the model_name or load a checkpoint.

    Parameters
    ----------
    config : Dict
        Config file dictionary
    """
    # TODO rewrite this ugly bullshit. registry,etc!
    model_name = config.algorithm.model
    load_checkpoint = config.algorithm.trained_model
    # TODO fix import
    # try:
    #     model_class = getattr(deconoising, model_name)
    # except ImportError:
    #     raise ImportError('Model not found')

    if model_name == "UNet":
        model = UNET(config.algorithm.conv_dims)
    if load_checkpoint:
        # TODO add proper logging message
        model.load_state_dict(torch.load(load_checkpoint))
        logger.info("Loaded checkpoint")
    return model
