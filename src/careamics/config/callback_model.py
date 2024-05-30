"""Callback Pydantic models."""

from __future__ import annotations

from datetime import timedelta
from typing import Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class CheckpointModel(BaseModel):
    """Checkpoint saving callback Pydantic model."""

    model_config = ConfigDict(
        validate_assignment=True,
    )

    monitor: Literal["val_loss"] = Field(default="val_loss", validate_default=True)
    verbose: bool = Field(default=False, validate_default=True)
    save_weights_only: bool = Field(default=False, validate_default=True)
    mode: Literal["min", "max"] = Field(default="min", validate_default=True)
    auto_insert_metric_name: bool = Field(default=False, validate_default=True)
    every_n_train_steps: Optional[int] = Field(
        default=None, ge=1, le=10, validate_default=True
    )
    train_time_interval: Optional[timedelta] = Field(
        default=None, validate_default=True
    )
    every_n_epochs: Optional[int] = Field(
        default=None, ge=1, le=10, validate_default=True
    )
    save_last: Optional[Literal[True, False, "link"]] = Field(
        default=True, validate_default=True
    )
    save_top_k: int = Field(default=3, ge=1, le=10, validate_default=True)


class EarlyStoppingModel(BaseModel):
    """Early stopping callback Pydantic model."""

    model_config = ConfigDict(
        validate_assignment=True,
    )

    monitor: Literal["val_loss"] = Field(default="val_loss", validate_default=True)
    patience: int = Field(default=3, ge=1, le=10, validate_default=True)
    mode: Literal["min", "max", "auto"] = Field(default="min", validate_default=True)
    min_delta: float = Field(default=0.0, ge=0.0, le=1.0, validate_default=True)
    check_finite: bool = Field(default=True, validate_default=True)
    stop_on_nan: bool = Field(default=True, validate_default=True)
    verbose: bool = Field(default=False, validate_default=True)
    restore_best_weights: bool = Field(default=True, validate_default=True)
    auto_lr_find: bool = Field(default=False, validate_default=True)
    auto_lr_find_patience: int = Field(default=3, ge=1, le=10, validate_default=True)
    auto_lr_find_mode: Literal["min", "max", "auto"] = Field(
        default="min", validate_default=True
    )
    auto_lr_find_direction: Literal["forward", "backward"] = Field(
        default="backward", validate_default=True
    )
    auto_lr_find_max_lr: float = Field(
        default=10.0, ge=0.0, le=1e6, validate_default=True
    )
    auto_lr_find_min_lr: float = Field(
        default=1e-8, ge=0.0, le=1e6, validate_default=True
    )
    auto_lr_find_num_training: int = Field(
        default=100, ge=1, le=1e6, validate_default=True
    )
    auto_lr_find_divergence_threshold: float = Field(
        default=5.0, ge=0.0, le=1e6, validate_default=True
    )
    auto_lr_find_accumulate_grad_batches: int = Field(
        default=1, ge=1, le=1e6, validate_default=True
    )
    auto_lr_find_stop_divergence: bool = Field(default=True, validate_default=True)
    auto_lr_find_step_scale: float = Field(default=0.1, ge=0.0, le=10)
