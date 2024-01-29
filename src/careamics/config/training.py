"""Training configuration."""
from __future__ import annotations

from typing import Dict, List

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from .filters import remove_default_optionals


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

    def model_dump(
        self, exclude_optionals: bool = True, *args: List, **kwargs: Dict
    ) -> Dict:
        """
        Override model_dump method.

        The purpose is to ensure export smooth import to yaml. It includes:
            - remove entries with None value.
            - remove optional values if they have the default value.

        Parameters
        ----------
        exclude_optionals : bool, optional
            Whether to exclude optional arguments if they are default, by default True.
        *args : List
            Positional arguments, unused.
        **kwargs : Dict
            Keyword arguments, unused.

        Returns
        -------
        dict
            Dictionary containing the model parameters.
        """
        dictionary = super().model_dump(exclude_none=True)

        if exclude_optionals:
            # remove optional arguments if they are default
            defaults = {
                "init_scale": 1024,
            }

            remove_default_optionals(dictionary, defaults)

        return dictionary


class Training(BaseModel):
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
        use_enum_values=True,
        validate_assignment=True,
    )

    # Mandatory fields
    num_epochs: int
    batch_size: int

    # Optional fields
    use_wandb: bool = False
    num_workers: int = Field(default=0, ge=0)
    amp: AMP = AMP()

    @field_validator("num_epochs", "batch_size")
    def greater_than_0(cls, val: int) -> int:
        """
        Validate number of epochs.

        Number of epochs must be greater than 0.

        Parameters
        ----------
        val : int
            Number of epochs.

        Returns
        -------
        int
            Validated number of epochs.

        Raises
        ------
        ValueError
            If the number of epochs is 0.
        """
        if val < 1:
            raise ValueError(f"Number of epochs must be greater than 0 (got {val}).")

        return val


    def model_dump(
        self, exclude_optionals: bool = True, *args: List, **kwargs: Dict
    ) -> Dict:
        """
        Override model_dump method.

        The purpose is to ensure export smooth import to yaml. It includes:
            - remove entries with None value.
            - remove optional values if they have the default value.

        Parameters
        ----------
        exclude_optionals : bool, optional
            Whether to exclude optional arguments if they are default, by default True.
        *args : List
            Positional arguments, unused.
        **kwargs : Dict
            Keyword arguments, unused.

        Returns
        -------
        dict
            Dictionary containing the model parameters.
        """
        dictionary = super().model_dump(exclude_none=True)

        if self.amp is not None:
            dictionary["amp"] = self.amp.model_dump(exclude_optionals)

        if exclude_optionals:
            # remove optional arguments if they are default
            defaults = {
                "use_wandb": False,
                "num_workers": 0,
                "amp": AMP().model_dump(),
            }

            remove_default_optionals(dictionary, defaults)

        return dictionary
