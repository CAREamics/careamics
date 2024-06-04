"""Logger supported by CAREamics."""

from careamics.utils import BaseEnum


class SupportedLogger(str, BaseEnum):
    """Available loggers."""

    WANDB = "wandb"
    TENSORBOARD = "tensorboard"
