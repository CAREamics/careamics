"""Callback Pydantic models."""

from __future__ import annotations

from datetime import timedelta
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class CheckpointModel(BaseModel):
    """Checkpoint saving callback Pydantic model.

    The parameters corresponds to those of
    `pytorch_lightning.callbacks.ModelCheckpoint`.

    See:
    https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html#modelcheckpoint
    """

    model_config = ConfigDict(validate_assignment=True, validate_default=True)

    monitor: Literal["val_loss"] | str | None = Field(default="val_loss")
    """Quantity to monitor, currently only `val_loss`."""

    verbose: bool = Field(default=False)
    """Verbosity mode."""

    save_weights_only: bool = Field(default=False)
    """When `True`, only the model's weights will be saved (model.save_weights)."""

    save_last: Literal[True, False, "link"] | None = Field(default=True)
    """When `True`, saves a last.ckpt copy whenever a checkpoint file gets saved."""

    save_top_k: int = Field(
        default=3,
        ge=-1,
        le=100,
    )
    """If `save_top_k == k, the best k models according to the quantity monitored
    will be saved. If `save_top_k == 0`, no models are saved. if `save_top_k == -1`,
    all models are saved."""

    mode: Literal["min", "max"] = Field(default="min")
    """One of {min, max}. If `save_top_k != 0`, the decision to overwrite the current
    save file is made based on either the maximization or the minimization of the
    monitored quantity. For 'val_acc', this should be 'max', for 'val_loss' this should
    be 'min', etc.
    """

    auto_insert_metric_name: bool = Field(default=False)
    """When `True`, the checkpoints filenames will contain the metric name."""

    every_n_train_steps: int | None = Field(default=None, ge=1, le=1000)
    """Number of training steps between checkpoints."""

    train_time_interval: timedelta | None = Field(default=None)
    """Checkpoints are monitored at the specified time interval."""

    every_n_epochs: int | None = Field(default=None, ge=1, le=100)
    """Number of epochs between checkpoints."""


class EarlyStoppingModel(BaseModel):
    """Early stopping callback Pydantic model.

    The parameters corresponds to those of
    `pytorch_lightning.callbacks.ModelCheckpoint`.

    See:
    https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html#lightning.pytorch.callbacks.EarlyStopping
    """

    model_config = ConfigDict(
        validate_assignment=True,
        validate_default=True,
    )

    monitor: Literal["val_loss"] = Field(default="val_loss")
    """Quantity to monitor."""

    min_delta: float = Field(default=0.0, ge=0.0, le=1.0)
    """Minimum change in the monitored quantity to qualify as an improvement, i.e. an
    absolute change of less than or equal to min_delta, will count as no improvement."""

    patience: int = Field(default=3, ge=1, le=10)
    """Number of checks with no improvement after which training will be stopped."""

    verbose: bool = Field(default=False)
    """Verbosity mode."""

    mode: Literal["min", "max", "auto"] = Field(default="min")
    """One of {min, max, auto}."""

    check_finite: bool = Field(default=True)
    """When `True`, stops training when the monitored quantity becomes `NaN` or
    `inf`."""

    stopping_threshold: float | None = Field(default=None)
    """Stop training immediately once the monitored quantity reaches this threshold."""

    divergence_threshold: float | None = Field(default=None)
    """Stop training as soon as the monitored quantity becomes worse than this
    threshold."""

    check_on_train_epoch_end: bool | None = Field(default=False)
    """Whether to run early stopping at the end of the training epoch. If this is
    `False`, then the check runs at the end of the validation."""

    log_rank_zero_only: bool = Field(default=False)
    """When set `True`, logs the status of the early stopping callback only for rank 0
    process."""
