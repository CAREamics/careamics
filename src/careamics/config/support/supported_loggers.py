"""Logger supported by CAREamics."""

from enum import StrEnum


class SupportedLogger(StrEnum):
    """Available loggers."""

    WANDB = "wandb"
    TENSORBOARD = "tensorboard"
