"""Normalization strategies models."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class MeanStdNormModel(BaseModel):
    """Standard normalization (mean/std)."""

    name: Literal["mean_std"] = "mean_std"


class NoNormModel(BaseModel):
    """No normalization applied."""

    name: Literal["none"] = "none"


class QuantileNormModel(BaseModel):
    """Quantile normalization."""

    name: Literal["quantile"] = "quantile"
    lower: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Lower quantile value for normalization.",
    )
    upper: float = Field(
        default=0.99,
        ge=0.0,
        le=1.0,
        description="Upper quantile value for normalization.",
    )

    @model_validator(mode="after")
    def check_bounds(self):
        """
        Check that the lower quantile is less than the upper quantile.

        Returns
        -------
        Self
            The normalized model.
        """
        if self.lower >= self.upper:
            raise ValueError("lower must be less than upper")
        return self
