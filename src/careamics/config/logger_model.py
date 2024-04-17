from pathlib import Path
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class WandbLoggerModel(BaseModel):
    """
    Weights & Biases logger parameters.

    Attributes
    ----------
    save_dir :
        Directory to save files.
    """

    model_config = ConfigDict(
        validate_assignment=True,
    )
    # discriminator used to distinguish between different logger types
    name: Literal["wandb"]
    save_dir: Union[str, Path] = Field(default=".")


class TensorboardLoggerModel(BaseModel):
    """
    Tensorboard logger parameters.

    Attributes
    ----------
    save_dir :
        Directory to save files.
    """

    model_config = ConfigDict(
        validate_assignment=True,
    )
    # discriminator used to distinguish between different logger types
    name: Literal["tensorboard"]
    save_dir: Union[str, Path] = Field(default=".")

