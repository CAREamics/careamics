import logging
from pathlib import Path

import torch

from ..config import Configuration
from ..config.algorithm import ModelName
from ..utils import set_logging
from .unet import UNet

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
    load_checkpoint = config.run_params.trained_model
    # TODO fix import
    # try:
    #     model_class = getattr(deconoising, model_name)
    # except ImportError:
    #     raise ImportError('Model not found')

    if model_name == ModelName.UNET:
        model = UNet(
            depth=config.algorithm.depth,
            conv_dim=config.algorithm.conv_dims,
            num_filter_base=config.algorithm.num_filter_base,
        )
    # TODO add more models or remove if
    if load_checkpoint:
        # TODO add proper logging message
        model.load_state_dict(torch.load(load_checkpoint))

        logger.info(f"Loaded model from {Path(load_checkpoint).name}")
    return model
