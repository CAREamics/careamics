"""Pydantic model for the max patch filter."""

from typing import Literal

from .filter_config import FilterConfig


class MaxFilterConfig(FilterConfig):
    """Pydantic model for the max patch filter."""

    name: Literal["max"] = "max"
    """Name of the filter."""

    threshold: float
    """Threshold for the minimum of the max-filtered patch."""
