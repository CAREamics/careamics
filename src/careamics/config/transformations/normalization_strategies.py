from typing import Literal, Optional

from pydantic import BaseModel, Field


class MeanStdNormModel(BaseModel):
    """Standard normalization (mean/std)."""

    name: Literal["mean_std"] = "mean_std"
    image_means: list[float] | None = None
    image_stds: list[float] | None = None
    target_means: list[float] | None = None
    target_stds: list[float] | None = None


class NoNormModel(BaseModel):
    """No normalization applied."""

    name: Literal["none"] = "none"


class QuantileNormModel(BaseModel):
    """Quantile normalization."""

    name: Literal["quantile"] = "quantile"
    lower: Optional[float] = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Lower quantile value for normalization.",
    )
    upper: Optional[float] = Field(
        default=0.99,
        ge=0.0,
        le=1.0,
        description="Upper quantile value for normalization.",
    )
