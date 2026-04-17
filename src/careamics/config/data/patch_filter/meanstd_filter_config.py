"""Pydantic model for the mean std patch filter."""

from typing import Literal

from .patch_filter_config import PatchFilterConfig


class MeanSTDPatchFilterConfig(PatchFilterConfig):
    """Pydantic model for the mean std patch filter."""

    name: Literal["mean_std"] = "mean_std"
    """Name of the filter."""

    mean_threshold: float
    """Minimum mean intensity required to keep a patch."""

    std_threshold: float | None = None
    """Minimum standard deviation required to keep a patch."""
