from enum import Enum

from pydantic import BaseModel, Field, validator


# python 3.11: https://docs.python.org/3/library/enum.html
class LossName(str, Enum):
    """Represents a loss function."""

    n2v = "n2v"
    pn2v = "pn2v"
    # TODO add all the others


class ModelName(str, Enum):
    """Represents a model."""

    unet = "UNet"
    # TODO add all the others


class OptimizerName(str, Enum):
    """Represents an optimizer."""

    adam = "Adam"
    # TODO add all the others


class SchedulerName(str, Enum):
    """Represents a learning rate schedule."""

    reduce_lr_on_plateau = "ReduceLROnPlateau"
    # TODO add all the others


# TODO finish configuration
# TODO different name? model?
class Algorithm(BaseModel):
    """Parameters related to the model architecture."""

    name: str
    loss: list[LossName]
    model: ModelName = Field(default=ModelName.unet)
    num_masked_pixels: int = Field(default=128, ge=1, le=1024)  # example: bounds
    patch_size: list[int] = Field(
        ..., min_items=2, max_items=3
    )  # example: min/max items

    @validator("num_masked_pixels")
    def validate_num_masked_pixels(cls, num):
        # example validation
        if num % 32 != 0:
            raise ValueError("num_masked_pixels must be a multiple of 32")

        return num

    class Config:
        use_enum_values = True  # make sure that enum are exported as str


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
            if "betas" not in parameters:
                raise ValueError("betas is required for Adam optimizer")
            if "eps" not in parameters:
                raise ValueError("eps is required for Adam optimizer")
            if "weight_decay" not in parameters:
                raise ValueError("weight_decay is required for Adam optimizer")
            if "amsgrad" not in parameters:
                raise ValueError("amsgrad is required for Adam optimizer")

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

        # TODO validate

        return parameters

    class Config:
        use_enum_values = True  # make sure that enum are exported as str


class Training(BaseModel):
    """Parameters related to the training."""

    num_epochs: int = Field(default=100, ge=0)
    learning_rate: float = Field(default=0.001, ge=0.0001, le=0.1)
    optimizer: Optimizer
    lr_scheduler: LrScheduler

    class Config:
        use_enum_values = True  # make sure that enum are exported as str


class Config(BaseModel):
    """Main configuration model."""

    experiment_name: str
    workdir: str
    algorithm: Algorithm
    training: Training
