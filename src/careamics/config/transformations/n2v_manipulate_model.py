from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class N2VManipulationParameters(BaseModel):
    """Pydantic model used to validate N2V manipulation parameters."""

    model_config = ConfigDict(
        validate_assignment=True,
    )

    roi_size: int = Field(default=11, ge=3, le=21)
    masked_pixel_percentage: float = Field(default=0.2, ge=0.05, le=1.0)
    strategy: Literal["uniform", "median"] = Field(default="uniform")
    struct_mask_axis: Literal["horizontal", "vertical", "none"] = Field(default="none")
    struct_mask_span: int = Field(default=5, ge=3, le=15)

    @field_validator("roi_size", "struct_mask_span")
    @classmethod
    def odd_value(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError("Size must be an odd number.")
        return v


class N2VManipulationModel(BaseModel):
    """Pydantic model used to represent N2V manipulation."""

    model_config = ConfigDict(
        validate_assignment=True,
    )

    name: Literal["N2VManipulate"]
    parameters: N2VManipulationParameters = N2VManipulationParameters()
