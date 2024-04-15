"""Training configuration."""
from __future__ import annotations

from pprint import pformat
from typing import Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from .callback_model import CheckpointModel, EarlyStoppingModel


# TODO: adapt for lightning:
# https://pytorch-lightning.readthedocs.io/en/1.8.6/api/pytorch_lightning.plugins.precision.PrecisionPlugin.html
class AMP(BaseModel):
    """
    Automatic mixed precision (AMP) parameters.

    See: https://pytorch.org/docs/stable/amp.html.

    Attributes
    ----------
    use : bool, optional
        Whether to use AMP or not, default False.
    init_scale : int, optional
        Initial scale used for loss scaling, default 1024.
    """

    model_config = ConfigDict(
        validate_assignment=True,
    )

    use: bool = False

    init_scale: int = Field(default=1024, ge=512, le=65536)

    @field_validator("init_scale")
    @classmethod
    def power_of_two(cls, scale: int) -> int:
        """
        Validate that init_scale is a power of two.

        Parameters
        ----------
        scale : int
            Initial scale used for loss scaling.

        Returns
        -------
        int
            Validated initial scale.

        Raises
        ------
        ValueError
            If the init_scale is not a power of two.
        """
        if not scale & (scale - 1) == 0:
            raise ValueError(f"Init scale must be a power of two (got {scale}).")

        return scale


class TrainingModel(BaseModel):
    """
    Parameters related to the training.

    Mandatory parameters are:
        - num_epochs: number of epochs, greater than 0.
        - batch_size: batch size, greater than 0.
        - augmentation: whether to use data augmentation or not (True or False).

    The other fields are optional:
        - use_wandb: whether to use wandb or not (default True).
        - num_workers: number of workers (default 0).
        - amp: automatic mixed precision parameters (disabled by default).

    Attributes
    ----------
    num_epochs : int
        Number of epochs, greater than 0.
    batch_size : int
        Batch size, greater than 0.
    augmentation : bool
        Whether to use data augmentation or not.
    use_wandb : bool
        Optional, whether to use wandb or not (default True).
    num_workers : int
        Optional, number of workers (default 0).
    amp : AMP
        Optional, automatic mixed precision parameters (disabled by default).
    """

    # Pydantic class configuration
    model_config = ConfigDict(
        validate_assignment=True,
    )

    # Mandatory fields
    num_epochs: int = Field(default=20, ge=1)

    # Optional fields
    # use_wandb: bool = False
    checkpoint_callback: CheckpointModel = CheckpointModel()
    early_stopping_callback: Optional[EarlyStoppingModel] = Field(
        default=None, validate_default=True
    )
    # precision: Literal["64", "32", "16", "bf16"] = 32

    def __str__(self) -> str:
        """Pretty string reprensenting the configuration.

        Returns
        -------
        str
            Pretty string.
        """
        return pformat(self.model_dump())
