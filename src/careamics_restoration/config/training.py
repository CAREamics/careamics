from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FieldValidationInfo,
    field_validator,
    model_validator,
)
from torch import optim

from .config_filter import remove_default_optionals
from .torch_optimizer import TorchLRScheduler, TorchOptimizer, get_parameters


class ExtractionStrategies(str, Enum):
    """Available extraction strategies.

    Currently supported:
        - random: random extraction.
        - sequential: grid extraction, can miss edge values.
        - tiled: tiled extraction, covers the whole image.
    """

    RANDOM = "random"
    SEQUENTIAL = "sequential"
    TILED = "tiled"


class Optimizer(BaseModel):
    """Torch optimizer.

    Only parameters supported by the corresponding torch optimizer will be taken
    into account. For more details, check:
    https://pytorch.org/docs/stable/optim.html#algorithms

    Note that mandatory parameters (see the specific Optimizer signature in the
    link above) must be provided. For example, SGD requires `lr`.

    Attributes
    ----------
    name : TorchOptimizer
        Name of the optimizer.
    parameters : dict
        Parameters of the optimizer (see torch documentation).
    """

    # Pydantic class configuration
    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
    )

    # Mandatory field
    name: TorchOptimizer

    # Optional parameters
    parameters: dict = {}

    @field_validator("parameters")
    def filter_parameters(cls, user_params: dict, values: FieldValidationInfo):
        """Validate optimizer parameters."""
        if "name" in values.data:
            optimizer_name = values.data["name"]

            # retrieve the corresponding optimizer class
            optimizer_class = getattr(optim, optimizer_name)

            # filter the user parameters according to the optimizer's signature
            return get_parameters(optimizer_class, user_params)
        else:
            raise ValueError(
                "Cannot validate optimizer parameters without `name`, check that it "
                "has correctly been specified."
            )

    @model_validator(mode="after")
    def sgd_lr_parameter(cls, optimizer: Optimizer) -> Optimizer:
        """Check that SGD optimizer as `lr` parameter specified.

        Parameters
        ----------
        optimizer : Optimizer
            Optimizer to validate.

        Returns
        -------
        Optimizer
            Validated optimizer.
        """
        if optimizer.name == TorchOptimizer.SGD and "lr" not in optimizer.parameters:
            raise ValueError(
                "SGD optimizer requires `lr` parameter, check that it has correctly "
                "been specified in `parameters`."
            )

        return optimizer

    def model_dump(self, exclude_optionals=True, *args, **kwargs) -> dict:
        """Override model_dump method.

        The purpose is to ensure export smooth import to yaml. It includes:
            - remove entries with None value
            - remove optional values if they have the default value

        Parameters
        ----------
        exclude_optionals : bool, optional
            Whether to exclude optional arguments if they are default, by default True.

        Returns
        -------
        dict
            Dictionary containing the model parameters
        """
        dictionary = super().model_dump(exclude_none=True)

        if exclude_optionals:
            # remove optional arguments if they are default
            default_optionals: dict = {"parameters": {}}

            remove_default_optionals(dictionary, default_optionals)

        return dictionary


class LrScheduler(BaseModel):
    """Torch learning rate scheduler.

    Only parameters supported by the corresponding torch lr scheduler will be taken
    into account. For more details, check:
    https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate

    Note that mandatory parameters (see the specific LrScheduler signature in the
    link above) must be provided. For example, StepLR requires `step_size`.

    Attributes
    ----------
    name : TorchLRScheduler
        Name of the learning rate scheduler.
    parameters : dict
        Parameters of the learning rate scheduler (see torch documentation).
    """

    # Pydantic class configuration
    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
    )

    # Mandatory field
    name: TorchLRScheduler

    # Optional parameters
    parameters: dict = {}

    @field_validator("parameters")
    def filter_parameters(cls, user_params: dict, values: FieldValidationInfo):
        """Validate lr scheduler parameters."""
        if "name" in values.data:
            lr_scheduler_name = values.data["name"]

            # retrieve the corresponding lr scheduler class
            lr_scheduler_class = getattr(optim.lr_scheduler, lr_scheduler_name)

            # filter the user parameters according to the lr scheduler's signature
            return get_parameters(lr_scheduler_class, user_params)
        else:
            raise ValueError(
                "Cannot validate lr scheduler parameters without `name`, check that it "
                "has correctly been specified."
            )

    @model_validator(mode="after")
    def step_lr_step_size_parameter(cls, lr_scheduler: LrScheduler) -> LrScheduler:
        """Check that StepLR lr scheduler has `step_size` parameter specified.

        Parameters
        ----------
        lr_scheduler : LrScheduler
            Lr scheduler to validate.

        Returns
        -------
        LrScheduler
            Validated lr scheduler.
        """
        if (
            lr_scheduler.name == TorchLRScheduler.StepLR
            and "step_size" not in lr_scheduler.parameters
        ):
            raise ValueError(
                "StepLR lr scheduler requires `step_size` parameter, check that it has "
                "correctly been specified in `parameters`."
            )

        return lr_scheduler

    def model_dump(self, exclude_optionals: bool = True, *args, **kwargs) -> dict:
        """Override model_dump method.

        The purpose is to ensure export smooth import to yaml. It includes:
            - remove entries with None value
            - remove optional values if they have the default value

        Parameters
        ----------
        exclude_optionals : bool, optional
            Whether to exclude optional arguments if they are default, by default True.

        Returns
        -------
        dict
            Dictionary containing the model parameters
        """
        dictionary = super().model_dump(exclude_none=True)

        if exclude_optionals:
            # remove optional arguments if they are default
            default_optionals: dict = {"parameters": {}}
            remove_default_optionals(dictionary, default_optionals)

        return dictionary


class AMP(BaseModel):
    """Automatic mixed precision (AMP) parameters.

    See:
    https://pytorch.org/docs/stable/amp.html


    Attributes
    ----------
    use : bool
        Whether to use AMP or not.
    init_scale : int
        Initial scale used for loss scaling.
    """

    model_config = ConfigDict(
        validate_assignment=True,
    )

    use: bool = False

    # Optional
    # TODO review init_scale and document better
    init_scale: int = Field(default=1024, ge=512, le=65536)

    @field_validator("init_scale")
    def power_of_two(cls, scale: int):
        """Validate that init_scale is a power of two."""
        if not scale & (scale - 1) == 0:
            raise ValueError(f"Init scale must be a power of two (got {scale}).")

        return scale

    def model_dump(self, exclude_optionals=True, *args, **kwargs) -> dict:
        """Override model_dump method.

        The purpose is to ensure export smooth import to yaml. It includes:
            - remove entries with None value
            - remove optional values if they have the default value

        Parameters
        ----------
        exclude_optionals : bool, optional
            Whether to exclude optional arguments if they are default, by default True.

        Returns
        -------
        dict
            Dictionary containing the model parameters
        """
        dictionary = super().model_dump(exclude_none=True)

        if exclude_optionals:
            # remove optional arguments if they are default
            defaults = {"init_scale": 1024}

            remove_default_optionals(dictionary, defaults)

        return dictionary


class Training(BaseModel):
    """Parameters related to the training.

    Mandatory parameters are:
        - num_epochs: number of epochs, greater than 0.
        - patch_size: patch size, 2D or 3D, non-zero and divisible by 2.
        - batch_size: batch size, greater than 0.
        - optimizer: optimizer, see `Optimizer`.
        - lr_scheduler: learning rate scheduler, see `LrScheduler`.
        - extraction_strategy: extraction strategy, see `ExtractionStrategies`.
        - augmentation: whether to use data augmentation or not (True or False).

    The other fields are optional:
        - use_wandb: whether to use wandb or not (default True).
        - num_workers: number of workers (default 0).
        - amp: automatic mixed precision parameters (disabled by default).

    Attributes
    ----------
    num_epochs : int
        Number of epochs, greater than 0.
    patch_size : conlist(int, min_length=2, max_length=3)
        Patch size, 2D or 3D, non-zero and divisible by 2.
    batch_size : int
        Batch size, greater than 0.
    optimizer : Optimizer
        Optimizer.
    lr_scheduler : LrScheduler
        Learning rate scheduler.
    extraction_strategy : ExtractionStrategies
        Extraction strategy.
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
    patch_size: List[int] = Field(..., min_length=2, max_length=3)
    batch_size: int

    optimizer: Optimizer
    lr_scheduler: LrScheduler

    extraction_strategy: ExtractionStrategies

    augmentation: bool

    # Optional fields
    use_wandb: bool = True
    num_workers: int = Field(default=0, ge=0)
    amp: AMP = AMP()

    @field_validator("num_epochs", "batch_size")
    def greater_than_0(cls, val: int) -> int:
        """Validate number of epochs.

        Number of epochs must be greater than 0.
        """
        if val < 1:
            raise ValueError(f"Number of epochs must be greater than 0 (got {val}).")

        return val

    @field_validator("patch_size")
    def all_elements_non_zero_divisible_by_2(cls, patch_list: List[int]) -> List[int]:
        """Validate patch size.

        Patch size must be non-zero, positive and divisible by 2.
        """
        for dim in patch_list:
            if dim < 1:
                raise ValueError(f"Patch size must be non-zero positive (got {dim}).")

            if dim % 2 != 0:
                raise ValueError(f"Patch size must be divisible by 2 (got {dim}).")

        return patch_list

    def model_dump(self, exclude_optionals=True, *args, **kwargs) -> dict:
        """Override model_dump method.

        The purpose is to ensure export smooth import to yaml. It includes:
            - remove entries with None value
            - remove optional values if they have the default value

        Parameters
        ----------
        exclude_optionals : bool, optional
            Whether to exclude optional arguments if they are default, by default True.

        Returns
        -------
        dict
            Dictionary containing the model parameters
        """
        dictionary = super().model_dump(exclude_none=True)

        dictionary["optimizer"] = self.optimizer.model_dump(exclude_optionals)
        dictionary["lr_scheduler"] = self.lr_scheduler.model_dump(exclude_optionals)

        if self.amp is not None:
            dictionary["amp"] = self.amp.model_dump(exclude_optionals)

        if exclude_optionals:
            # remove optional arguments if they are default
            defaults = {
                "use_wandb": True,
                "num_workers": 0,
                "amp": AMP().model_dump(),
            }

            remove_default_optionals(dictionary, defaults)

        return dictionary
