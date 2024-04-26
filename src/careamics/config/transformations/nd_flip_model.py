"""Pydantic model for the NDFlip transform."""
from typing import Literal

from pydantic import ConfigDict, Field

from .transform_model import TransformModel


class NDFlipModel(TransformModel):
    """
    Pydantic model used to represent NDFlip transformation.

    Attributes
    ----------
    name : Literal["NDFlip"]
        Name of the transformation.
    p : float
        Probability of applying the transformation, by default 0.5.
    is_3D : bool
        Whether the transformation should be applied in 3D, by default False.
    flip_z : bool
        Whether to flip the z axis, by default True.
    """

    model_config = ConfigDict(
        validate_assignment=True,
    )

    name: Literal["NDFlip"]
    p: float = Field(default=0.5, ge=0.0, le=1.0)
    is_3D: bool = Field(default=False)
    flip_z: bool = Field(default=True)
