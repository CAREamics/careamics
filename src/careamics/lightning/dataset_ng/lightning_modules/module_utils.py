"""Training utilities for Lightning modules."""

from typing import Any

import pytorch_lightning as L
import torch
from torchmetrics import MetricCollection

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


def load_best_checkpoint(module: L.LightningModule) -> bool:
    """Load the best checkpoint from the trainer's checkpoint callback.

    Parameters
    ----------
    module : L.LightningModule
        The Lightning module to load the checkpoint into.

    Returns
    -------
    bool
        True if checkpoint was loaded, False otherwise.
    """
    if (
        not hasattr(module.trainer, "checkpoint_callback")
        or module.trainer.checkpoint_callback is None
    ):
        logger.warning("No checkpoint callback found, cannot load best checkpoint.")
        return False

    best_model_path = module.trainer.checkpoint_callback.best_model_path  # type: ignore[attr-defined]
    if best_model_path and best_model_path != "":
        logger.info(f"Loading best checkpoint from: {best_model_path}")
        model_state = torch.load(best_model_path, weights_only=True)["state_dict"]
        module.load_state_dict(model_state)
        return True
    else:
        logger.warning("No best checkpoint found.")
        return False
