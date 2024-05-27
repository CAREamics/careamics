"""Pydantic model for the NDFlip transform."""

from typing import Literal, Optional

from pydantic import ConfigDict

from .transform_model import TransformModel


class NDFlipModel(TransformModel):
    """
    Pydantic model used to represent NDFlip transformation.

    Attributes
    ----------
    name : Literal["NDFlip"]
        Name of the transformation.
    seed : Optional[int]
        Seed for the random number generator.
    """

    model_config = ConfigDict(
        validate_assignment=True,
    )

    name: Literal["NDFlip"] = "NDFlip"
    seed: Optional[int] = None
