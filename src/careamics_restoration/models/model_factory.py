from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch

from ..bioimage import import_bioimage_model
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
    if model_name == Models.UNET:
        return UNet
    else:
        raise NotImplementedError(f"Model {model_name} is not implemented")


def create_model(
    *,
    model_path: Optional[Union[str, Path]] = None,
    config: Optional[Configuration] = None,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """Creates a model from a configuration file or a checkpoint.

    One of `config` or `model_path` must be provided. If both are provided, only
    `model_path` is used.

    Parameters
    ----------
    model_path : Optional[Union[str, Path]], optional
        Path to a checkpoint, by default None
    config : Optional[Configuration], optional
        Configuration object, by default None

    Returns
    -------
    torch.nn.Module
        Model object


    Raises
    ------
    ValueError
        If neither config nor model_path is provided
    """
    if model_path is not None:
        # Create model from checkpoint
        model_path = Path(model_path)
        if not model_path.exists() or model_path.suffix not in [".pth", ".zip"]:
            raise ValueError(
                f"Invalid model path: {model_path}. Current working dir: \
                              {Path.cwd()!s}"
            )

        if model_path.suffix == ".zip":
            model_path = import_bioimage_model(model_path)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        # Load the configuration
        if "config" in checkpoint:
            config = Configuration(**checkpoint["config"])
            algo_config = config.algorithm
            model_config = algo_config.model_parameters
            model_name = algo_config.model
        else:
            raise ValueError("Invalid checkpoint format, no configuration found.")

        # Create model
        model = model_registry(model_name)(
            depth=model_config.depth,
            conv_dim=algo_config.get_conv_dim(),
            num_channels_init=model_config.num_channels_init,
        )
        model.to(device)
        # Load the model state dict
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Loaded model state dict")
        else:
            raise ValueError("Invalid checkpoint format")

        # Load the optimizer and scheduler
        optimizer, scheduler = get_optimizer_and_scheduler(
            config, model, state_dict=checkpoint
        )
        scaler = get_grad_scaler(config, state_dict=checkpoint)

    elif config is not None:
        # Create model from configuration
        algo_config = config.algorithm
        model_config = algo_config.model_parameters
        model_name = algo_config.model

        # Create model
        model = model_registry(model_name)(
            depth=model_config.depth,
            conv_dim=algo_config.get_conv_dim(),
            num_channels_init=model_config.num_channels_init,
        )
        model.to(device)
        assert config is not None, "Configuration must be provided"  # mypy
        optimizer, scheduler = get_optimizer_and_scheduler(config, model)
        scaler = get_grad_scaler(config)
        logger.info("Engine initialized from configuration")

    else:
        raise ValueError("Either config or model_path must be provided")

    return model, optimizer, scheduler, scaler, config


def get_optimizer_and_scheduler(
    config: Configuration, model: torch.nn.Module, state_dict: Optional[Dict] = None
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Creates optimizer and learning rate scheduler objects.

    Parameters
    ----------
    config : Configuration
        Configuration object
    model : torch.nn.Module
        Model object
    state_dict : Optional[Dict], optional
        State dict of the checkpoint, by default None

    Returns
    -------
    Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]

    Raises
    ------
    ValueError
        If the entry is missing in the configuration file.
    """
    if config.training is not None:
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
    else:
        raise ValueError("Missing training entry in configuration file.")


def get_grad_scaler(
    cfg: Configuration, state_dict: Optional[Dict] = None
) -> torch.cuda.amp.GradScaler:
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
