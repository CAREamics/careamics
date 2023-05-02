from enum import Enum
from pathlib import Path

from typing import Optional, List
from pydantic import BaseModel, Field, validator

from n2v.dataloader_utils import are_axes_valid


class SupportedExtension(str, Enum):
    TIFF = "tiff"
    TIF = "tif"

    @classmethod
    def _missing_(cls, value):
        """Override default behaviour for missing values.

        Convert value to lowercase and try to match it with enum values.
        """
        lower_value = value.lower()

        # attempt to match lowercase value with enum values
        for member in cls:
            if member.value == lower_value:
                return member

        return super()._missing_(value)


class ExtractionStrategy(str, Enum):
    SEQUENTIAL = "sequential"


class Data(BaseModel):
    path: Path
    axes: str

    # optional with default values (included in yml)
    ext: SupportedExtension = SupportedExtension.TIF
    extraction_strategy: ExtractionStrategy = ExtractionStrategy.SEQUENTIAL

    batch_size: int = Field(default=1, ge=1)
    patch_size: List[int] = Field(..., min_items=2, max_items=3)

    # TODO add proper validation and augmentation parameters, list of strings ?

    # optional with None default values (not included in yml if not defined)
    num_files: Optional[int] = Field(None, ge=1)
    num_patches: Optional[int] = Field(None, ge=1)
    num_workers: Optional[int] = Field(default=None, ge=0, le=8)  # TODO is this used?

    # TODO how to make parameters mutually exclusive (which one???)

    @validator("patch_size")
    def validate_parameters(cls, patch_size: List[int]) -> List[int]:
        for p in patch_size:
            # check if power of 2 and divisible by 8
            if not (p & (p - 1) == 0) or p % 8 != 0:
                raise ValueError(
                    f"Patch size {p} is not a power of 2 or not divisible by 8"
                )
        return patch_size

    @validator("axes")
    def validate_axes(cls, axes, values, **kwargs):
        # validate axes
        are_axes_valid(axes)

        # check if comaptible with patch size
        if "patch_size" in values:
            if len(axes) != len(values["patch_size"]):
                raise ValueError(
                    f"Number of axes ({len(axes)}) and patch size "
                    f"({len(values['patch_size'])}) do not match."
                )

        return axes

    def dict(self, *args, **kwargs) -> dict:
        """Override dict method.

        The purpose is to ensure export smooth import to yaml. It includes:
            - remove entries with None value
            - replace Path by str
        """
        dictionary = super().dict(exclude_none=True)

        # replace Path by str
        dictionary["path"] = str(dictionary["path"])

        return dictionary

    class Config:
        use_enum_values = True  # enum are exported as str
        allow_mutation = False  # model is immutable
