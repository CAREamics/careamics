"""Training utilities for Lightning modules."""

from typing import Any

import lightning.pytorch as L
import torch
from torch import nn
from torchmetrics import MetricCollection

from careamics.config.support import SupportedOptimizer, SupportedScheduler
from careamics.utils.logging import get_logger

logger = get_logger(__name__)


def log_training_stats(module: L.LightningModule, loss: Any, batch_size: int) -> None:
    """Log training loss and learning rate.

    Parameters
    ----------
    module : L.LightningModule
        The Lightning module to log stats for.
    loss : Any
        The loss value for the current training step.
    batch_size : int
        The size of the batch used in the current training step.
    """
    module.log(
        "train_loss",
        loss,
        on_step=True,
        on_epoch=True,
        prog_bar=True,
        logger=True,
        batch_size=batch_size,
    )

    optimizer = module.optimizers()
    if isinstance(optimizer, list):
        current_lr = optimizer[0].param_groups[0]["lr"]
    else:
        current_lr = optimizer.param_groups[0]["lr"]
    module.log(
        "learning_rate",
        current_lr,
        on_step=False,
        on_epoch=True,
        logger=True,
        batch_size=batch_size,
    )


def log_validation_stats(
    module: L.LightningModule,
    loss: Any,
    batch_size: int,
    metrics: MetricCollection,
) -> None:
    """Log validation loss and metrics.

    Parameters
    ----------
    module : L.LightningModule
        The Lightning module to log stats for.
    loss : Any
        The loss value for the current validation step.
    batch_size : int
        The size of the batch used in the current validation step.
    metrics : MetricCollection
        The metrics collection to log.
    """
    module.log(
        "val_loss",
        loss,
        on_step=False,
        on_epoch=True,
        prog_bar=True,
        logger=True,
        batch_size=batch_size,
    )
    module.log_dict(metrics, on_step=False, on_epoch=True, batch_size=batch_size)


def log_images(
    module: L.LightningModule,
    input_or_target: torch.Tensor,
    prediction: torch.Tensor,
    is_supervised: bool,
) -> None:
    """Log input/target, and prediction images to the logger.

    Parameters
    ----------
    module : L.LightningModule
        The Lightning module to log images for.
    input_or_target : torch.Tensor
        The input or target images.
    prediction : torch.Tensor
        The predicted images.
    is_supervised : bool
        Whether the task is supervised (i.e., target images are available).
    """
    # denormalize images before logging
    normalization = module.trainer.datamodule.train_dataset.normalization  # type: ignore[union-attr]
    # log images to the logger
    tag_key = "Target" if is_supervised else "Input"
    module.logger.log_images(  # type: ignore
        key=tag_key,
        images=normalization.denormalize(input_or_target).cpu().numpy(),
        step=module.global_step,
        captions=[tag_key] * len(input_or_target),
    )
    module.logger.log_images(  # type: ignore
        key="Prediction",
        images=normalization.denormalize(prediction).cpu().numpy(),
        step=module.global_step,
        captions=["Prediction"] * len(prediction),
    )


def get_optimizer(name: str) -> type[torch.optim.Optimizer]:
    """
    Return the optimizer class given its name.

    Parameters
    ----------
    name : str
        Optimizer name.

    Returns
    -------
    torch.nn.Optimizer
        Optimizer class.
    """
    try:
        SupportedOptimizer(name)
    except ValueError as e:
        raise NotImplementedError(f"Optimizer {name} is not yet supported.") from e

    return getattr(torch.optim, name)


def get_scheduler(
    name: str,
) -> type[torch.optim.lr_scheduler.ReduceLROnPlateau]:
    """
    Return the scheduler class given its name.

    Parameters
    ----------
    name : str
        Scheduler name.

    Returns
    -------
    Union
        Scheduler class.
    """
    try:
        SupportedScheduler(name)
    except ValueError as e:
        raise NotImplementedError(f"Scheduler {name} is not yet supported.") from e

    return getattr(torch.optim.lr_scheduler, name)


def configure_optimizers(
    model: nn.Module,
    optimizer_name: str,
    optimizer_parameters: dict[str, Any],
    lr_scheduler_name: str,
    lr_scheduler_parameters: dict[str, Any],
    monitor: str = "val_loss",
) -> dict[str, Any]:
    """Configure optimizer and learning rate scheduler.

    Parameters
    ----------
    model : nn.Module
        The model whose parameters will be optimized.
    optimizer_name : str
        The name of the optimizer to use.
    optimizer_parameters : dict[str, Any]
        Parameters to pass to the optimizer constructor.
    lr_scheduler_name : str
        The name of the learning rate scheduler to use.
    lr_scheduler_parameters : dict[str, Any]
        Parameters to pass to the learning rate scheduler constructor.
    monitor : str, optional
        The metric to monitor for the learning rate scheduler, by default "val_loss".

    Returns
    -------
    dict[str, Any]
        A dictionary containing the optimizer and learning rate scheduler configuration.
    """
    optimizer_func = get_optimizer(optimizer_name)
    optimizer = optimizer_func(  # type: ignore[operator]
        model.parameters(), **optimizer_parameters
    )

    scheduler_func = get_scheduler(lr_scheduler_name)
    scheduler = scheduler_func(optimizer, **lr_scheduler_parameters)  # type: ignore[operator]

    return {
        "optimizer": optimizer,
        "lr_scheduler": scheduler,
        "monitor": monitor,
    }
