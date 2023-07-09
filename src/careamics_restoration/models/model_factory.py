from pathlib import Path

import torch

from ..config import Configuration
from ..config.algorithm import Models
from .unet import UNet
from ..utils.logging import get_logger

logger = get_logger(__name__)


def create_model(config: Configuration) -> torch.nn.Module:
    """Builds a model based on the model_name or load a checkpoint.

    Parameters
    ----------
    config : Dict
        Config file dictionary
    """
    # TODO rewrite this ugly bullshit. registry,etc!
    algo_config = config.algorithm
    model_config = algo_config.model_parameters

    model_name = algo_config.model
    load_checkpoint = config.trained_model

    # TODO fix import
    # try:
    #     model_class = getattr(deconoising, model_name)
    # except ImportError:
    #     raise ImportError('Model not found')

    if model_name == Models.UNET:
        model = UNet(
            depth=model_config.depth,
            conv_dim=algo_config.get_conv_dim(),
            num_filter_base=model_config.num_filters_base,
        )

    # TODO add more models or remove if
    if load_checkpoint is not None:
        # TODO add proper logging message
        if not Path(load_checkpoint).is_absolute():
            try:
                logger.info(
                    f"Trying to load checkpoint from relative path {config.working_directory / load_checkpoint}"
                )
                model.load_state_dict(
                    torch.load(config.working_directory / load_checkpoint)
                )
                logger.info(f"Loaded model from {Path(load_checkpoint).name}")
            except FileNotFoundError:
                raise FileNotFoundError(f"Checkpoint {load_checkpoint} not found")
        else:
            try:
                logger.info(
                    f"Trying to load checkpoint from absolute path {load_checkpoint}"
                )
                model.load_state_dict(torch.load(load_checkpoint))
                logger.info(f"Loaded model from {Path(load_checkpoint).name}")
            except FileNotFoundError:
                raise FileNotFoundError(f"Checkpoint {load_checkpoint} not found")
    return model
