"""Pydantic model for the Normalize transform."""

from typing import Literal

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
    mean: float = Field(default=0.485)  # albumentations defaults
    std: float = Field(default=0.229)
