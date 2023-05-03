from enum import Enum

from typing import List, Dict, Union
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
    loss: List[LossName]
    pixel_manipulation: str  # TODO same as name ?
    model: ModelName = Field(default=ModelName.unet)
    depth: int = Field(default=3, ge=2)  # example: bounds
    num_masked_pixels: int = Field(default=128, ge=1, le=1024)  # example: bounds
    conv_mult: int = Field(default=2, ge=2, le=3)  # example: bounds
    checkpoint: str = Field(default=None)

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


class Data(BaseModel):
    path: str
    ext: str = Field(default=".tif")  # TODO add regexp for list of extensions or enum
    axes: str
    num_files: Union[int, None] = Field(default=None)
    extraction_strategy: str = Field(default="sequential")  # TODO add enum
    patch_size: List[int] = Field(
        ..., min_items=2, max_items=3
    )  # TODO how to validate list
    num_patches: Union[int, None]  # TODO how to make parameters mutually exclusive
    batch_size: int
    num_workers: int = Field(default=0)
    # augmentation: None  # TODO add proper validation and augmentation parameters, list of strings ?

    @validator("patch_size")
    def validate_parameters(cls, patch_size):
        for p in patch_size:
            # TODO validate ,power of 2, divisible by 8 ? Should be acceptable for the model
            pass
        return patch_size

    @validator("axes")
    def validate_axes(cls, axes):
        # TODO validate axes, No C
        return axes


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
    running_stats: bool
    data: Data

    class Config:
        use_enum_values = True  # make sure that enum are exported as str


class Evaluation(BaseModel):
    data: Data
    metric: str  # TODO add enum


class Prediction(BaseModel):
    data: Data
    overlap: List[int] = Field(
        ..., min_items=2, max_items=3
    )  # TODO ge image size, check consistency with image size


class Stage(str, Enum):
    TRAINING = "training"
    EVALUATION = "evaluation"
    PREDICTION = "prediction"


class ConfigValidator(BaseModel):
    """Main configuration model."""

    experiment_name: str
    workdir: str
    algorithm: Algorithm
    training: Training
    evaluation: Evaluation
    prediction: Prediction

    def get_stage_config(self, stage: Union[str, Stage]) -> Union[Training, Evaluation]:
        if stage == Stage.TRAINING:
            return self.training
        elif stage == Stage.EVALUATION:
            return self.evaluation
        else:
            return self.prediction
