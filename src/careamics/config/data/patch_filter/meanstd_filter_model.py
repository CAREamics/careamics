"""Pydantic model for the mean std patch filter."""

from typing import Literal

from .filter_model import FilterModel


class MeanSTDFilterModel(FilterModel):
    """Pydantic model for the mean std patch filter."""

    name: Literal["mean_std"] = "mean_std"
    """Name of the filter."""

    mean_threshold: float
    """Minimum mean intensity required to keep a patch."""

    std_threshold: float | None = None
    """Minimum standard deviation required to keep a patch."""
