from enum import Enum

from pydantic import BaseModel, Field, validator


# python 3.11: https://docs.python.org/3/library/enum.html
class Loss(str, Enum):
    n2v = "n2v"
    pn2v = "pn2v"
    # TODO add all the others


class Model(str, Enum):
    unet = "UNet"
    # TODO add all the others


# TODO finish configuration
class Algorithm(BaseModel):
    name: str
    loss: list[Loss]
    model: Model
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


class Config(BaseModel):
    experiment_name: str
    workdir: str
    algorithm: Algorithm
