"""Training configuration."""

from __future__ import annotations

from pprint import pformat
from typing import Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

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

    num_epochs: int = Field(default=20, ge=1)
    """Number of epochs, greater than 0."""

    precision: Literal['64', '32', '16-mixed', 'bf16-mixed'] = Field(default='32')
    """Numerical precision"""s
    max_steps: int = Field(default=-1, ge=-1, exclude=0)
    """Maximum number of steps to train for. -1 means no limit."""
    check_val_every_n_epoch: Optional[int] = Field(default=1, ge=1)
    """Validation step frequency."""
    enable_progress_bar: Optional[bool] = Field(default=True)
    """Whether to enable the progress bar."""
    accumulate_grad_batches: int = 1,
    """Number of batches to accumulate gradients over before stepping the optimizer."""
    gradient_clip_val: Optional[Union[int, float]] = None,
    """The value to which to clip the gradient"""
    gradient_clip_algorithm: Optional[Literal["value", "norm"]] = None,
    """The algorithm to use for gradient clipping (see lightning `Trainer`)."""
    logger: Optional[Literal["wandb", "tensorboard"]] = None
    """Logger to use during training. If None, no logger will be used. Available
    loggers are defined in SupportedLogger."""

    checkpoint_callback: CheckpointModel = CheckpointModel()
    """Checkpoint callback configuration, following PyTorch Lightning Checkpoint
    callback."""

    early_stopping_callback: Optional[EarlyStoppingModel] = Field(
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
