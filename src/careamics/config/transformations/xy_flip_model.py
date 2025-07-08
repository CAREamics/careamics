"""Pydantic model for the XYFlip transform."""

from typing import Literal

from pydantic import ConfigDict, Field

from .transform_model import TransformModel


class XYFlipModel(TransformModel):
    """
    Pydantic model used to represent XYFlip transformation.

    Attributes
    ----------
    name : Literal["XYFlip"]
        Name of the transformation.
    p : float
        Probability of applying the transform, by default 0.5.
    seed : Optional[int]
        Seed for the random number generator,  by default None.
    """

    model_config = ConfigDict(
        validate_assignment=True,
    )

    name: Literal["XYFlip"] = "XYFlip"
    flip_x: bool = Field(
        True,
        description="Whether to flip along the X axis.",
    )
    flip_y: bool = Field(
        True,
        description="Whether to flip along the Y axis.",
    )
    p: float = Field(
        0.5,
        description="Probability of applying the transform.",
        ge=0,
        le=1,
    )
    seed: int | None = None
