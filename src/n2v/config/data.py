from enum import Enum

from typing import List, Union
from pydantic import BaseModel, Field, validator


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

    class Config:
        allow_mutation = False  # model is immutable
