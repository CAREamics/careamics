from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class NDFlipParameters(BaseModel):
    """Pydantic model used to validate NDFlip parameters."""

    model_config = ConfigDict(
        validate_assignment=True,
    )

    p: float = Field(default=0.5, ge=0.0, le=1.0)
    is_3D: bool = Field(default=False)
    flip_z: bool = Field(default=True)


class NDFlipModel(BaseModel):
    """Pydantic model used to represent NDFlip transformation."""

    model_config = ConfigDict(
        validate_assignment=True,
    )

    name: Literal["NDFlip"]
    parameters: NDFlipParameters = NDFlipParameters()
