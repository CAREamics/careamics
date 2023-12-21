"""
Model factory.

Model creation factory functions.
"""
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch

from careamics.bioimage import import_bioimage_model
from careamics.config import Configuration
from careamics.config.algorithm import Models
from careamics.utils.logging import get_logger

from .unet import UNet

logger = get_logger(__name__)


def model_registry(model_name: str) -> torch.nn.Module:
    """
    Model factory.

    Supported models are defined in config.algorithm.Models.

    Parameters
    ----------
    model_name : str
        Name of the model.

    Returns
    -------
    torch.nn.Module
        Model class.

    Raises
    ------
    NotImplementedError
        If the requested model is not implemented.
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
) -> Tuple[
    torch.nn.Module,
    torch.optim.Optimizer,
    Union[
        torch.optim.lr_scheduler.LRScheduler,
        torch.optim.lr_scheduler.ReduceLROnPlateau,  # not a subclass of LRScheduler
    ],
    torch.cuda.amp.GradScaler,
    Configuration,
]:
    """
    Instantiate a model from a model path or configuration.

    If both path and configuration are provided, the model path is used. The model
    path should point to either a checkpoint (created during training) or a model
    exported to the bioimage.io format.

    Parameters
    ----------
    model_path : Optional[Union[str, Path]], optional
        Path to a checkpoint or bioimage.io archive, by default None.
    config : Optional[Configuration], optional
        Configuration, by default None.
    device : Optional[torch.device], optional
        Torch device, by default None.

    Returns
    -------
    torch.nn.Module
        Instantiated model.

    Raises
    ------
    ValueError
        If the checkpoint path is invalid.
    ValueError
        If the checkpoint is invalid.
    ValueError
        If neither checkpoint nor configuration are provided.
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
        model: torch.nn.Module = model_registry(model_name)(
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
        optimizer, scheduler = get_optimizer_and_scheduler(config, model)
        scaler = get_grad_scaler(config)
        logger.info("Engine initialized from configuration")

    else:
        raise ValueError("Either config or model_path must be provided")

    return model, optimizer, scheduler, scaler, config


def get_optimizer_and_scheduler(
    config: Configuration, model: torch.nn.Module, state_dict: Optional[Dict] = None
) -> Tuple[
    torch.optim.Optimizer,
    Union[
        torch.optim.lr_scheduler.LRScheduler,
        torch.optim.lr_scheduler.ReduceLROnPlateau,  # not a subclass of LRScheduler
    ],
]:
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
