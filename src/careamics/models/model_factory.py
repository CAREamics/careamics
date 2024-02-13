"""
Model factory.

Model creation factory functions.
"""
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch

from ..bioimage import import_bioimage_model
from ..config import Configuration
from ..config.architectures import UNetModel
from ..config.support import SupportedArchitecture
from ..utils.logging import get_logger
from .unet import UNet

logger = get_logger(__name__)


# TODO rename model factory
def model_registry(model_configuration: UNetModel) -> torch.nn.Module:
    """
    Model factory.

    Supported models are defined in careamics.config.architectures.Architectures.

    Parameters
    ----------
    model_configuration : UNetModel
        Model configuration

    Returns
    -------
    torch.nn.Module
        Model class.

    Raises
    ------
    NotImplementedError
        If the requested model is not implemented.
    """
    if model_configuration.architecture == SupportedArchitecture.UNET:
        return UNet(
            **dict(model_configuration)
        )
    else:
        raise NotImplementedError(
            f"Model {model_configuration.architecture} is not implemented or unknown."
        )


def get_optimizer_and_scheduler(
    config: Configuration, model: torch.nn.Module, state_dict: Optional[Dict] = None
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """
    Create optimizer and learning rate schedulers.

    If a checkpoint state dictionary is provided, the optimizer and scheduler are
    instantiated to the same state as the checkpoint's optimizer and scheduler.

    Parameters
    ----------
    config : Configuration
        Configuration.
    model : torch.nn.Module
        Model.
    state_dict : Optional[Dict], optional
        Checkpoint state dictionary, by default None.

    Returns
    -------
    Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]
        Optimizer and scheduler.
    """
    # retrieve optimizer name and parameters from config
    optimizer_name = config.training.optimizer.name
    optimizer_params = config.training.optimizer.parameters

    # then instantiate it
    optimizer_func = getattr(torch.optim, optimizer_name)
    optimizer = optimizer_func(model.parameters(), **optimizer_params)

    # same for learning rate scheduler
    scheduler_name = config.training.lr_scheduler.name
    scheduler_params = config.training.lr_scheduler.parameters
    scheduler_func = getattr(torch.optim.lr_scheduler, scheduler_name)
    scheduler = scheduler_func(optimizer, **scheduler_params)

    # load state from ther checkpoint if available
    if state_dict is not None:
        if "optimizer_state_dict" in state_dict:
            optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            logger.info("Loaded optimizer state dict")
        else:
            logger.warning(
                "No optimizer state dict found in checkpoint. Optimizer not loaded."
            )
        if "scheduler_state_dict" in state_dict:
            scheduler.load_state_dict(state_dict["scheduler_state_dict"])
            logger.info("Loaded LR scheduler state dict")
        else:
            logger.warning(
                "No LR scheduler state dict found in checkpoint. "
                "LR scheduler not loaded."
            )
    return optimizer, scheduler


def get_grad_scaler(
    config: Configuration, state_dict: Optional[Dict] = None
) -> torch.cuda.amp.GradScaler:
    """
    Instantiate gradscaler.

    If a checkpoint state dictionary is provided, the scaler is instantiated to the
    same state as the checkpoint's scaler.

    Parameters
    ----------
    config : Configuration
        Configuration.
    state_dict : Optional[Dict], optional
        Checkpoint state dictionary, by default None.

    Returns
    -------
    torch.cuda.amp.GradScaler
        Instantiated gradscaler.
    """
    use = config.training.amp.use
    scaling = config.training.amp.init_scale
    scaler = torch.cuda.amp.GradScaler(init_scale=scaling, enabled=use)
    if state_dict is not None and "scaler_state_dict" in state_dict:
        scaler.load_state_dict(state_dict["scaler_state_dict"])
        logger.info("Loaded GradScaler state dict")
    return scaler
