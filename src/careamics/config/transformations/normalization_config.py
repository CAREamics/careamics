"""Pydantic model for the Normalize transform."""

from typing import Literal

from pydantic import ConfigDict, Field, BaseModel


class StandardizeModel(BaseModel):
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

    name: Literal["standard"] = "standard"
    image_means: list = Field(..., min_length=0, max_length=32)
    image_stds: list = Field(..., min_length=0, max_length=32)
    target_means: list | None = Field(default=None, min_length=0, max_length=32)
    target_stds: list | None = Field(default=None, min_length=0, max_length=32)


class NoNormModel(BaseModel):
    """Pydantic model used to represent no normalization."""

    name: Literal["none"] = "none"


class QuantileModel(BaseModel):
    """Pydantic model used to represent quantile normalization."""

    name: Literal["quantile"] = "quantile"
    lower_quantiles: list = Field(..., min_length=0, max_length=32)
    upper_quantiles: list = Field(..., min_length=0, max_length=32)


class MinMaxModel(BaseModel):
    """Pydantic model used to represent min-max normalization."""

    name: Literal["minmax"] = "minmax"
    image_mins: list = Field(..., min_length=0, max_length=32)
    image_maxs: list = Field(..., min_length=0, max_length=32)
