"""Pydantic model for the XYFlip transform."""

from typing import Literal, Optional

from pydantic import ConfigDict

from .transform_model import TransformModel


class XYFlipModel(TransformModel):
    """
    Pydantic model used to represent XYFlip transformation.

    Attributes
    ----------
    name : Literal["XYFlip"]
        Name of the transformation.
    seed : Optional[int]
        Seed for the random number generator.
    """

    model_config = ConfigDict(
        validate_assignment=True,
    )

    name: Literal["XYFlip"] = "XYFlip"
    seed: Optional[int] = None
