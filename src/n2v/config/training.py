from enum import Enum

from pydantic import BaseModel, Field, validator

from .data import Data


class OptimizerName(str, Enum):
    """Represents an optimizer."""

    adam = "Adam"
    # TODO add all the others


class SchedulerName(str, Enum):
    """Represents a learning rate schedule."""

    reduce_lr_on_plateau = "ReduceLROnPlateau"
    # TODO add all the others


class Optimizer(BaseModel):
    """Parameters related to the optimizer."""

    name: OptimizerName
    parameters: dict

    # validate parameters using value of name
    @validator("parameters")
    def validate_parameters(cls, parameters, values):
        name = values["name"]

        # TODO: check types and values of the parameters
        # TODO: alternatively we can do a pydantic model of each optimizer
        if name == OptimizerName.adam:
            if "lr" not in parameters:
                raise ValueError("lr is required for Adam optimizer")

        return parameters

    class Config:
        use_enum_values = True  # make sure that enum are exported as str


class LrScheduler(BaseModel):
    """Parameters related to the learning rate scheduler."""

    name: SchedulerName
    parameters: dict

    # validate parameters using value of name
    @validator("parameters")
    def validate_parameters(cls, parameters, values):
        name = values["name"]

        if name == SchedulerName.reduce_lr_on_plateau:
            if "mode" not in parameters:
                raise ValueError("mode is required for ReduceLROnPlateau scheduler")
            if "factor" not in parameters:
                raise ValueError("factor is required for ReduceLROnPlateau scheduler")
            if "patience" not in parameters:
                raise ValueError("patience is required for ReduceLROnPlateau scheduler")

        return parameters

    class Config:
        use_enum_values = True  # make sure that enum are exported as str


class Amp(BaseModel):
    toggle: bool
    init_scale: int  # TODO excessive ?


class Training(BaseModel):
    """Parameters related to the training."""

    num_epochs: int = Field(default=100, ge=0)
    learning_rate: float = Field(default=0.001, ge=0.0001, le=0.1)
    optimizer: Optimizer
    lr_scheduler: LrScheduler
    amp: Amp
    max_grad_norm: float = Field(default=1.0, ge=0.0, le=1.0)
    data: Data

    class Config:
        use_enum_values = True  # enum are exported as str
        allow_mutation = False  # model is immutable
