"""Pydantic model for the Normalize transform."""

from typing import List, Literal

from pydantic import ConfigDict, Field

from .transform_model import TransformModel


class NormalizeModel(TransformModel):
    """
    Pydantic model used to represent Normalize transformation.

    The Normalize transform is a zero mean and unit variance transformation.

    Attributes
    ----------
    name : Literal["Normalize"]
        Name of the transformation.
    mean : float
        Mean value for normalization.
    std : float
        Standard deviation value for normalization.
    """

    model_config = ConfigDict(
        validate_assignment=True,
    )

    name: Literal["Normalize"] = "Normalize"
    image_means: List[float] = Field(default=[], min_length=0, max_length=32)
    image_stds: List[float] = Field(default=[], min_length=0, max_length=32)
    target_means: List[float] = Field(default=[], min_length=0, max_length=32)
    target_stds: List[float] = Field(default=[], min_length=0, max_length=32)

