"""Pydantic model for the XYRandomRotate90 transform."""

from typing import Literal, Optional

from pydantic import ConfigDict, Field

from .transform_model import TransformModel


class XYRandomRotate90Model(TransformModel):
    """
    Pydantic model used to represent the XY random 90 degree rotation transformation.

    Attributes
    ----------
    name : Literal["XYRandomRotate90"]
        Name of the transformation.
    p : float
        Probability of applying the transform, by default 0.5.
    seed : Optional[int]
        Seed for the random number generator, by default None.
    """

    model_config = ConfigDict(
        validate_assignment=True,
    )

    name: Literal["XYRandomRotate90"] = "XYRandomRotate90"
    p: float = Field(
        0.5,
        description="Probability of applying the transform.",
        ge=0,
        le=1,
    )
    seed: Optional[int] = None
