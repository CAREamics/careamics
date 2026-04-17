"""Pydantic model for the max patch filter."""

from typing import Literal

from pydantic import Field

from .patch_filter_config import PatchFilterConfig


class MaxPatchFilterConfig(PatchFilterConfig):
    """Pydantic model for the max patch filter."""

    name: Literal["max"] = "max"
    """Name of the filter."""

    threshold: float
    """Threshold for the minimum of the max-filtered patch."""

    coverage: float = Field(0.25, ge=0.0, le=1.0)
    """Minimum ratio of masked pixels required to keep a sampling region. The optimum
    value is 1/(2**ndims) where ndims is the number of spatial dimensions.
    """
