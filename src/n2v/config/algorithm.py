from enum import Enum

from typing import List
from pydantic import BaseModel, Field, validator


# python 3.11: https://docs.python.org/3/library/enum.html
class LossName(str, Enum):
    """Class representing a loss function.

    Accepted losses are defined in loss.py."""

    n2v = "n2v"
    pn2v = "pn2v"


class ModelName(str, Enum):
    """Class representing a model.

    Accepted models are defined in model.py."""

    unet = "UNet"


class Algorithm(BaseModel):
    """Algorithm configuration model.

    Attributes
    ----------
    name : str
        Name of the algorithm
    """

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
        use_enum_values = True  # enum are exported as str
        allow_mutation = False  # model is immutable
