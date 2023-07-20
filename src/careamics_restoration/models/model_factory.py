from pathlib import Path
from typing import Tuple

import torch

from ..config import Configuration
from ..config.algorithm import Models
from ..utils.logging import get_logger
from .unet import UNet

logger = get_logger(__name__)


def model_registry(model_name: str) -> torch.nn.Module:
    """Returns a dictionary of models. WIP.

    Returns
    -------
    model
    """
    # TODO make this a dict, not a function
    # TODO: add more models
    if model_name == Models.UNET:
        return UNet
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")


def create_model(config: Configuration) -> torch.nn.Module:
    """Builds a model based on the model_name or load a checkpoint.

    Parameters
    ----------
    config : Dict
        Config file dictionary
    """
    algo_config = config.algorithm
    model_config = algo_config.model_parameters
    model_name = algo_config.model

    load_checkpoint = config.trained_model
    model = model_registry(model_name)(
        depth=model_config.depth,
        conv_dim=algo_config.get_conv_dim(),
        num_channels_init=model_config.num_channels_init,
    )
    if load_checkpoint is not None:
        checkpoint = torch.load(load_checkpoint)
        if "model_state_dict" in checkpoint:
            logger.info("Trying to load model state dict")
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded model from {Path(load_checkpoint).name}")
        else:
            # TODO: add jit, onxx, etc.
            raise ValueError("Invalid checkpoint format")

        logger.info(f"Loaded model from {Path(load_checkpoint).name}")
        if "config" in checkpoint:
            config.data.mean = checkpoint["config"]["data"]["mean"]
            config.data.std = checkpoint["config"]["data"]["std"]
            logger.info("Updated config from checkpoint")
            # TODO discuss other updates
    optimizer, scheduler = get_optimizer_and_scheduler(
        config, model, state_dict=checkpoint if load_checkpoint else None
    )
    scaler = get_grad_scaler(config, state_dict=checkpoint if load_checkpoint else None)
    return model, optimizer, scheduler, scaler, config


def get_optimizer_and_scheduler(
    cfg, model, state_dict=None
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Creates optimizer and learning rate scheduler objects.

    Returns
    -------
    Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]

    Raises
    ------
    ValueError
        If the entry is missing in the configuration file.
    """
    if cfg.training is not None:
        # retrieve optimizer name and parameters from config
        optimizer_name = cfg.training.optimizer.name
        optimizer_params = cfg.training.optimizer.parameters

        # then instantiate it
        optimizer_func = getattr(torch.optim, optimizer_name)
        optimizer = optimizer_func(model.parameters(), **optimizer_params)

        # same for learning rate scheduler
        scheduler_name = cfg.training.lr_scheduler.name
        scheduler_params = cfg.training.lr_scheduler.parameters
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
    else:
        raise ValueError("Missing training entry in configuration file.")


def get_grad_scaler(cfg, state_dict=None) -> torch.cuda.amp.GradScaler:
    """Create the gradscaler object.

    Returns
    -------
    torch.cuda.amp.GradScaler

    Raises
    ------
    ValueError
        If the entry is missing in the configuration file.
    """
    if cfg.training is not None:
        use = cfg.training.amp.use
        scaling = cfg.training.amp.init_scale
        scaler = torch.cuda.amp.GradScaler(init_scale=scaling, enabled=use)
        if state_dict is not None and "scaler_state_dict" in state_dict:
            scaler.load_state_dict(state_dict["scaler_state_dict"])
            logger.info("Loaded GradScaler state dict")
        return scaler
    else:
        raise ValueError("Missing training entry in configuration file.")
