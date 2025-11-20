"""Pydantic model for the mask coordinate filter."""

from typing import Literal

from pydantic import Field

from .filter_config import FilterConfig


class MaskFilterConfig(FilterConfig):
    """Pydantic model for the mask coordinate filter."""

    name: Literal["mask"] = "mask"
    """Name of the filter."""

    coverage: float = Field(0.5, ge=0.0, le=1.0)
    """Percentage of masked pixels required to keep a patch."""
