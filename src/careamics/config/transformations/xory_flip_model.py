"""Pydantic model for the XorYFlip transform."""

from typing import Literal, Optional

from pydantic import ConfigDict, Field

from .transform_model import TransformModel


class XorYFlipModel(TransformModel):
    """
    Pydantic model used to represent XorYFlip transformation.

    Attributes
    ----------
    name : Literal["XYFlip"]
        Name of the transformation.
    axis : Literal[-2, -1]
        Axis to be flipped.
    p : float
        Probability of applying the transform, by default 0.5.
    seed : Optional[int]
        Seed for the random number generator,  by default None.
    """

    model_config = ConfigDict(
        validate_assignment=True,
    )

    name: Literal["XYFlip"] = "XYFlip"
    axis: Literal[-2, -1]
    p: float = Field(
        0.5,
        description="Probability of applying the transform.",
        ge=0,
        le=1,
    )
    seed: Optional[int] = None
