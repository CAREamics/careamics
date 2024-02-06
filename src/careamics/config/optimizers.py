from __future__ import annotations

from typing import Dict, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from torch import optim

from careamics.utils.torch_utils import get_parameters

from .support.supported_optimizers import SupportedOptimizer, SupportedScheduler


class OptimizerModel(BaseModel):
    """
    Torch optimizer.

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
        validate_assignment=True,
    )

    # Mandatory field
    name: Literal["Adam", "SGD"] = Field(default="Adam", validate_default=True)

    # Optional parameters, empty dict default value to allow filtering dictionary
    parameters: dict = Field(default={}, validate_default=True)

    @field_validator("name", mode="before")
    def validate_name(cls, name: str) -> str:
        """
        Validate optimizer name.

        Parameters
        ----------
        name : str
            Name of the optimizer.

        Returns
        -------
        str
            Validated optimizer name.

        Raises
        ------
        ValueError
            If the optimizer name is not supported.
        """
        if name != "custom" and name not in SupportedOptimizer.__members__.values():
            raise ValueError(
                f"Optimizer {name} is not supported, check that it has correctly "
                "been specified."
            )

        return name

    @field_validator("parameters", mode="before")
    def filter_parameters(cls, user_params: dict, values: ValidationInfo) -> Dict:
        """
        Validate optimizer parameters.

        This method filters out unknown parameters, given the optimizer name.

        Parameters
        ----------
        user_params : dict
            Parameters passed on to the torch optimizer.
        values : ValidationInfo
            Pydantic field validation info, used to get the optimizer name.

        Returns
        -------
        Dict
            Filtered optimizer parameters.

        Raises
        ------
        ValueError
            If the optimizer name is not specified.
        """
        # None value to default
        if user_params is None:
            user_params = {}

        # since we are validating before type validation, enforce is here
        if not isinstance(user_params, dict):
            raise ValueError(
                f"Optimizer parameters must be a dictionary, got {type(user_params)}."
            )

        if "name" in values.data:  # TODO test if that clause if really necessary
            optimizer_name = values.data["name"]

            # retrieve the corresponding optimizer class
            optimizer_class = getattr(optim, optimizer_name)

            # filter the user parameters according to the optimizer's signature
            #TODO this don't store full list of defaults
            return get_parameters(optimizer_class, user_params) # TODO now return a tuple
        else:
            raise ValueError(
                "Cannot validate optimizer parameters without `name`, check that it "
                "has correctly been specified."
            )

    @model_validator(mode="after")
    def sgd_lr_parameter(cls, optimizer: OptimizerModel) -> OptimizerModel:
        """
        Check that SGD optimizer has the mandatory `lr` parameter specified.

        Parameters
        ----------
        optimizer : Optimizer
            Optimizer to validate.

        Returns
        -------
        Optimizer
            Validated optimizer.

        Raises
        ------
        ValueError
            If the optimizer is SGD and the lr parameter is not specified.
        """
        if (
            optimizer.name == SupportedOptimizer.SGD
            and "lr" not in optimizer.parameters
        ):
            raise ValueError(
                "SGD optimizer requires `lr` parameter, check that it has correctly "
                "been specified in `parameters`."
            )

        return optimizer


class LrSchedulerModel(BaseModel):
    """
    Torch learning rate scheduler.

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
        validate_assignment=True,
    )

    # Mandatory field
    name: str = Field(default="ReduceLROnPlateau")

    # Optional parameters
    parameters: dict = Field(default={}, validate_default=True)

    @field_validator("name", mode="before")
    def validate_name(cls, name: str) -> str:
        """
        Validate lr scheduler name.

        Parameters
        ----------
        name : str
            Name of the lr scheduler.

        Returns
        -------
        str
            Validated lr scheduler name.

        Raises
        ------
        ValueError
            If the lr scheduler name is not supported.
        """
        if name != "custom" and name not in SupportedScheduler.__members__.values():
            raise ValueError(
                f"Lr scheduler {name} is not supported, check that it has correctly "
                "been specified."
            )

        return name

    @field_validator("parameters", mode="before")
    def filter_parameters(cls, user_params: dict, values: ValidationInfo) -> Dict:
        """
        Validate lr scheduler parameters.

        This method filters out unknown parameters, given the lr scheduler name.

        Parameters
        ----------
        user_params : dict
            Parameters passed on to the torch lr scheduler.
        values : ValidationInfo
            Pydantic field validation info, used to get the lr scheduler name.

        Returns
        -------
        Dict
            Filtered lr scheduler parameters.

        Raises
        ------
        ValueError
            If the lr scheduler name is not specified.
        """
        # None value to default
        if user_params is None:
            user_params = {}

        # since we are validating before type validation, enforce is here
        if not isinstance(user_params, dict):
            raise ValueError(
                f"Optimizer parameters must be a dictionary, got {type(user_params)}."
            )

        if "name" in values.data:
            lr_scheduler_name = values.data["name"]

            # retrieve the corresponding lr scheduler class
            lr_scheduler_class = getattr(optim.lr_scheduler, lr_scheduler_name)

            # filter the user parameters according to the lr scheduler's signature
            return get_parameters(lr_scheduler_class, user_params) # TODO now return a tuple
        else:
            raise ValueError(
                "Cannot validate lr scheduler parameters without `name`, check that it "
                "has correctly been specified."
            )

    @model_validator(mode="after")
    def step_lr_step_size_parameter(
        cls, lr_scheduler: LrSchedulerModel
    ) -> LrSchedulerModel:
        """
        Check that StepLR lr scheduler has `step_size` parameter specified.

        Parameters
        ----------
        lr_scheduler : LrScheduler
            Lr scheduler to validate.

        Returns
        -------
        LrScheduler
            Validated lr scheduler.

        Raises
        ------
        ValueError
            If the lr scheduler is StepLR and the step_size parameter is not specified.
        """
        if (
            lr_scheduler.name == SupportedScheduler.StepLR
            and "step_size" not in lr_scheduler.parameters
        ):
            raise ValueError(
                "StepLR lr scheduler requires `step_size` parameter, check that it has "
                "correctly been specified in `parameters`."
            )

        return lr_scheduler
