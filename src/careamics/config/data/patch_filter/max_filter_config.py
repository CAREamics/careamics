"""Pydantic model for the max patch filter."""

from typing import Literal

from pydantic import Field

from .filter_config import FilterConfig


class MaxFilterConfig(FilterConfig):
    """Pydantic model for the max patch filter."""

    name: Literal["max"] = "max"
    """Name of the filter."""

    threshold: float
    """Threshold for the minimum of the max-filtered patch."""

    coverage: float | None = Field(None, ge=0.0, le=1.0)
    """Minimum ratio of pixels greater than the threshold required to keep a sampling 
    region, `default=None`. If `None` then `1/(2**ndims)` is used where `ndims` is the 
    number of spatial dimensions.
    """
