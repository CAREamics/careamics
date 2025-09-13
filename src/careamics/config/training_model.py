"""Training configuration."""

from __future__ import annotations

from pprint import pformat
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .callback_model import CheckpointModel, EarlyStoppingModel


class TrainingConfig(BaseModel):
    """
    Parameters related to the training.

    Mandatory parameters are:
        - num_epochs: number of epochs, greater than 0.
        - batch_size: batch size, greater than 0.
        - augmentation: whether to use data augmentation or not (True or False).

    Attributes
    ----------
    num_epochs : int
        Number of epochs, greater than 0.
    """

    # Pydantic class configuration
    model_config = ConfigDict(
        validate_assignment=True,
    )
    lightning_trainer_config: dict | None = None
    """Configuration for the PyTorch Lightning Trainer, following PyTorch Lightning
    Trainer class"""

    logger: Literal["wandb", "tensorboard"] | None = None
    """Logger to use during training. If None, no logger will be used. Available
    loggers are defined in SupportedLogger."""

    # Only basic callbacks
    checkpoint_callback: CheckpointModel = CheckpointModel()
    """Checkpoint callback configuration, following PyTorch Lightning Checkpoint
    callback."""

    early_stopping_callback: EarlyStoppingModel | None = Field(
        default=None, validate_default=True
    )
    """Early stopping callback configuration, following PyTorch Lightning Checkpoint
    callback."""

    def __str__(self) -> str:
        """Pretty string reprensenting the configuration.

        Returns
        -------
        str
            Pretty string.
        """
        return pformat(self.model_dump())

    def has_logger(self) -> bool:
        """Check if the logger is defined.

        Returns
        -------
        bool
            Whether the logger is defined or not.
        """
        return self.logger is not None
