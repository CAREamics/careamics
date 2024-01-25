"""Convenience functions to instantiate torch.optim optimizers and schedulers."""
from __future__ import annotations

import inspect
from enum import Enum
from typing import Dict, List


from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationInfo,
    model_validator,
    field_validator
)
from torch import optim

from .filters import remove_default_optionals


class TorchOptimizer(str, Enum):
    """
    Supported optimizers.

    Currently only supports Adam and SGD.
    """

    # ASGD = "ASGD"
    # Adadelta = "Adadelta"
    # Adagrad = "Adagrad"
    Adam = "Adam"
    # AdamW = "AdamW"
    # Adamax = "Adamax"
    # LBFGS = "LBFGS"
    # NAdam = "NAdam"
    # RAdam = "RAdam"
    # RMSprop = "RMSprop"
    # Rprop = "Rprop"
    SGD = "SGD"
    # SparseAdam = "SparseAdam"


class TorchLRScheduler(str, Enum):
    """
    Supported learning rate schedulers.

    Currently only supports ReduceLROnPlateau and StepLR.
    """

    # ChainedScheduler = "ChainedScheduler"
    # ConstantLR = "ConstantLR"
    # CosineAnnealingLR = "CosineAnnealingLR"
    # CosineAnnealingWarmRestarts = "CosineAnnealingWarmRestarts"
    # CyclicLR = "CyclicLR"
    # ExponentialLR = "ExponentialLR"
    # LambdaLR = "LambdaLR"
    # LinearLR = "LinearLR"
    # MultiStepLR = "MultiStepLR"
    # MultiplicativeLR = "MultiplicativeLR"
    # OneCycleLR = "OneCycleLR"
    # PolynomialLR = "PolynomialLR"
    ReduceLROnPlateau = "ReduceLROnPlateau"
    # SequentialLR = "SequentialLR"
    StepLR = "StepLR"




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
        use_enum_values=True,
        validate_assignment=True,
    )

    # Mandatory field
    name: TorchOptimizer

    # Optional parameters, empty dict default value to allow filtering dictionary
    parameters: dict = {}

    @field_validator("parameters", mode='before')
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
        if optimizer.name == TorchOptimizer.SGD and "lr" not in optimizer.parameters:
            raise ValueError(
                "SGD optimizer requires `lr` parameter, check that it has correctly "
                "been specified in `parameters`."
            )

        return optimizer

    def model_dump(
        self, exclude_optionals: bool = True, *args: List, **kwargs: Dict
    ) -> Dict:
        """
        Override model_dump method.

        The purpose of this method is to ensure smooth export to yaml. It
        includes:
            - removing entries with None value.
            - removing optional values if they have the default value.

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
            default_optionals: dict = {"parameters": {}}

            remove_default_optionals(dictionary, default_optionals)

        return dictionary


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
        use_enum_values=True,
        validate_assignment=True,
    )

    # Mandatory field
    name: TorchLRScheduler

    # Optional parameters
    parameters: dict = {}

    @field_validator("parameters", mode='before')
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
            return get_parameters(lr_scheduler_class, user_params)
        else:
            raise ValueError(
                "Cannot validate lr scheduler parameters without `name`, check that it "
                "has correctly been specified."
            )

    @model_validator(mode="after")
    def step_lr_step_size_parameter(cls, lr_scheduler: LrSchedulerModel) -> LrSchedulerModel:
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
            lr_scheduler.name == TorchLRScheduler.StepLR
            and "step_size" not in lr_scheduler.parameters
        ):
            raise ValueError(
                "StepLR lr scheduler requires `step_size` parameter, check that it has "
                "correctly been specified in `parameters`."
            )

        return lr_scheduler

    def model_dump(
        self, exclude_optionals: bool = True, *args: List, **kwargs: Dict
    ) -> Dict:
        """
        Override model_dump method.

        The purpose of this method is to ensure smooth export to yaml. It includes:
            - removing entries with None value.
            - removing optional values if they have the default value.

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
            default_optionals: dict = {"parameters": {}}
            remove_default_optionals(dictionary, default_optionals)

        return dictionary




def get_parameters(
    func: type,
    user_params: dict,
) -> dict:
    """
    Filter parameters according to the function signature.

    Parameters
    ----------
    func : type
        Class object.
    user_params : Dict
        User provided parameters.

    Returns
    -------
    Dict
        Parameters matching `func`'s signature.
    """
    # Get the list of all default parameters
    default_params = list(inspect.signature(func).parameters.keys())

    # Filter matching parameters
    params_to_be_used = set(user_params.keys()) & set(default_params)

    return {key: user_params[key] for key in params_to_be_used}


def get_optimizers() -> Dict[str, str]:
    """
    Return the list of all optimizers available in torch.optim.

    Returns
    -------
    Dict
        Optimizers available in torch.optim.
    """
    optims = {}
    for name, obj in inspect.getmembers(optim):
        if inspect.isclass(obj) and issubclass(obj, optim.Optimizer):
            if name != "Optimizer":
                optims[name] = name
    return optims


def get_schedulers() -> Dict[str, str]:
    """
    Return the list of all schedulers available in torch.optim.lr_scheduler.

    Returns
    -------
    Dict
        Schedulers available in torch.optim.lr_scheduler.
    """
    schedulers = {}
    for name, obj in inspect.getmembers(optim.lr_scheduler):
        if inspect.isclass(obj) and issubclass(obj, optim.lr_scheduler.LRScheduler):
            if "LRScheduler" not in name:
                schedulers[name] = name
        elif name == "ReduceLROnPlateau":  # somewhat not a subclass of LRScheduler
            schedulers[name] = name
    return schedulers
