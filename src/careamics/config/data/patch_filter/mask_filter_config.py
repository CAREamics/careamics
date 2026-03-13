"""Pydantic model for the mask coordinate filter."""

from typing import Literal

from pydantic import BaseModel, Field


class MaskFilterConfig(BaseModel):
    """Pydantic model for the mask coordinate filter."""

    name: Literal["mask"] = "mask"
    """Name of the filter."""

    coverage: float | None = Field(None, ge=0.0, le=1.0)
    """Minimum ratio of masked pixels required to keep a sampling region, 
    `default=None`. If `None` then `1/(ndims**2)` is used where `ndims` is the number of
    spatial dimensions.
    """
