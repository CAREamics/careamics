from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class NormalizeParameters(BaseModel):
    """Pydantic model used to validate Normalize parameters.
    """

    model_config = ConfigDict(
        validate_assignment=True,
    )

    mean: float = Field(default=0.485) # albumentations default
    std: float = Field(default=0.229)
    max_pixel_value: float = Field(default=1.0, ge=0.0) # TODO explain why


class NormalizeModel(BaseModel):
    """Pydantic model used to represent Normalize transformation.
    """

    model_config = ConfigDict(
        validate_assignment=True,
    )

    name: Literal["Normalize"]
    parameters: NormalizeParameters = NormalizeParameters()
    