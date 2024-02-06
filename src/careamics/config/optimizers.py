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

from careamics.utils.torch_utils import filter_parameters


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

    @field_validator("parameters")
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
        optimizer_name = values.data["name"]

        # retrieve the corresponding optimizer class
        optimizer_class = getattr(optim, optimizer_name)

        # filter the user parameters according to the optimizer's signature
        parameters, missing_mandatory = filter_parameters(optimizer_class, user_params)

        # remove `params` from missing mandatory parameters, see torch.optim docs
        missing_mandatory = [
            p for p in missing_mandatory if p != "params"
        ]

        # if there are missing parameters, raise an error
        if len(missing_mandatory) > 0:
            raise ValueError(
                f"Optimizer {optimizer_name} requires the following parameters: "
                f"{missing_mandatory}."
            )

        return parameters
    

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
    name: Literal["ReduceLROnPlateau", "StepLR"] = Field(default="ReduceLROnPlateau")

    # Optional parameters
    parameters: dict = Field(default={}, validate_default=True)

    @field_validator("parameters")
    def filter_parameters(cls, user_params: dict, values: ValidationInfo) -> Dict:

        scheduler_name = values.data["name"]

        # retrieve the corresponding scheduler class
        scheduler_class = getattr(optim.lr_scheduler, values.data["name"])

        # filter the user parameters according to the scheduler's signature
        parameters, missing_mandatory = filter_parameters(scheduler_class, user_params)

        # remove `optimizer` from missing mandatory parameters
        # see torch.optim.lr_scheduler docs
        missing_mandatory = [
            p for p in missing_mandatory if p != "optimizer"
        ]

        # if there are missing parameters, raise an error
        if len(missing_mandatory) > 0:
            raise ValueError(
                f"Optimizer {scheduler_name} requires the following parameters: "
                f"{missing_mandatory}."
            )

        return parameters
    
