"""Pydantic model for the Shannon entropy patch filter."""

from typing import Literal

from .filter_model import FilterModel


class ShannonFilterModel(FilterModel):
    """Pydantic model for the Shannon entropy patch filter."""

    name: Literal["shannon"] = "shannon"
    """Name of the filter."""

    threshold: float
    """Minimum Shannon entropy required to keep a patch."""
