"""Pydantic model for the XYRandomRotate90 transform."""
from typing import Literal

from pydantic import ConfigDict, Field

from .transform_model import TransformModel


class XYRandomRotate90Model(TransformModel):
    """
    Pydantic model used to represent NDFlip transformation.

    Attributes
    ----------
    name : Literal["XYRandomRotate90"]
        Name of the transformation.
    p : float
        Probability of applying the transformation, by default 0.5.
    is_3D : bool
        Whether the transformation should be applied in 3D, by default False.
    """

    model_config = ConfigDict(
        validate_assignment=True,
    )

    name: Literal["XYRandomRotate90"] = "XYRandomRotate90"
    p: float = Field(default=0.5, ge=0.0, le=1.0)
    is_3D: bool = Field(default=False)
