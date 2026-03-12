"""Training configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pprint import pformat
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


@dataclass
class SupervisedCheckpointing:
    """Presets for checkpointing CARE.

    This preset saves the top 3 best performing checkpoints based on `val_loss`, as well
    as the last one.
    """

    monitor: str = "val_loss"
    """Monitor `val_loss`."""

    mode: str = "min"
    """Top checkpoints are selected by minimum `val_loss`."""

    save_top_k: int = 3
    """Save the top 3 best performing checkpoints."""

    save_last: bool = True
    """Save the last checkpoint."""

    auto_insert_metric_name: bool = False
    """Do not insert the monitored value in the checkpoint name."""


@dataclass
class SelfSupervisedCheckpointing:
    """Presets for checkpointing Noise2Noise and Noise2Void.

    Because self-supervised algorithms are evaluating the loss against noisy pixels,
    its value is not a good measure of performances after a few epochs. Therefore, it
    cannot be used to evaluate the best performing models.

    This presets saves checkpoints every 10 epochs, as well as the last one.
    """

    every_n_epochs: int = 10
    """Save a checkpoint every 10 epochs."""

    save_top_k: int = -1
    """Do not save the checkpoints based on a monitored value."""

    save_last: bool = True
    """Save the last checkpoint."""

    auto_insert_metric_name: bool = False
    """Do not insert the monitored value in the checkpoint name."""


class NGTrainingConfig(BaseModel):
    """
    Parameters related to the training.

    Attributes
    ----------
    lightning_trainer_config : dict
        Configuration for the PyTorch Lightning Trainer, following PyTorch Lightning
        Trainer class.
    logger : Literal["wandb", "tensorboard"] | None
        Additional Logger to use during training. If None, no logger will be used.
        Note that the `CAREamist` uses the `csv` logger regardless of the value of this
        field.
    checkpoint_callback : dict[str, Any]
        Checkpoint callback configuration, following PyTorch Lightning Checkpoint
        callback.
    early_stopping_callback : dict[str, Any] | None
        Early stopping callback configuration, following PyTorch Lightning Checkpoint
        callback.
    """

    # Pydantic class configuration
    model_config = ConfigDict(
        validate_assignment=True,
    )

    lightning_trainer_config: dict = Field(default={})
    """Configuration for the PyTorch Lightning Trainer, following PyTorch Lightning
    Trainer class"""

    logger: Literal["wandb", "tensorboard"] | None = None
    """Logger to use during training. If None, no logger will be used. Available
    loggers are defined in SupportedLogger."""

    # Only basic callbacks - they may have different defaults for different algorithms
    checkpoint_callback: dict[str, Any] = Field(default_factory=dict)
    """Checkpoint callback configuration, following PyTorch Lightning Checkpoint
    callback."""

    early_stopping_callback: dict[str, Any] | None = Field(default_factory=dict)
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
