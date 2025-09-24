"""Pydantic model for the max patch filter."""

from typing import Literal

from .filter_model import FilterModel


class MaxFilterModel(FilterModel):
    """Pydantic model for the max patch filter."""

    name: Literal["max"] = "max"
    """Name of the filter."""

    threshold: float
    """Threshold for the minimum of the max-filtered patch."""
