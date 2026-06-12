"""Pydantic model for the mask patch filter."""

from typing import Literal

from pydantic import Field

from .patch_filter_config import PatchFilterConfig


class MaskPatchFilterConfig(PatchFilterConfig):
    """Pydantic model for the mask patch filter."""

    name: Literal["mask"] = "mask"
    """Name of the filter."""

    coverage: float = Field(0.25, ge=0.0, le=1.0)
    """Minimum ratio of masked pixels required to keep a sampling region. The optimum
    value is 1/(2**ndims) where ndims is the number of spatial dimensions.
    """
