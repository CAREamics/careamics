"""Pydantic model for the XorYFlip transform."""

from typing import Literal, Optional

from pydantic import ConfigDict, Field

from .transform_model import TransformModel


class XorYFlipModel(TransformModel):
    """Pydantic model representing a single-axis (X or Y) flip transformation.

    Attributes
    ----------
    name : Literal["XorYFlip"]
        Name of the transformation.
    flip_x : bool
        If True, flip along the X axis, otherwise flip along the Y axis.
    p : float
        Probability of applying the transform, by default 0.5.
    seed : Optional[int]
        Seed for the random number generator,  by default None.
    """

    model_config = ConfigDict(
        validate_assignment=True,
    )

    name: Literal["XorYFlip"] = "XorYFlip"
    flip_x: bool = Field(
        ...,
        description="If True, flip along the X axis, otherwise flip along the Y axis.",
    )
    p: float = Field(
        0.5,
        description="Probability of applying the transform.",
        ge=0,
        le=1,
    )
    seed: Optional[int] = None
