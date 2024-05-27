"""Pydantic model for the XYRandomRotate90 transform."""

from typing import Literal, Optional

from pydantic import ConfigDict

from .transform_model import TransformModel


class XYRandomRotate90Model(TransformModel):
    """
    Pydantic model used to represent NDFlip transformation.

    Attributes
    ----------
    name : Literal["XYRandomRotate90"]
        Name of the transformation.
    seed : Optional[int]
        Seed for the random number generator.
    """

    model_config = ConfigDict(
        validate_assignment=True,
    )

    name: Literal["XYRandomRotate90"] = "XYRandomRotate90"
    seed: Optional[int] = None
