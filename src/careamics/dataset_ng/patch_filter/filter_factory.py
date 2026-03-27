"""Factories for coordinate and patch filters."""

from typing import Union

from careamics.config.data.patch_filter import (
    FilterConfig,
    MaxFilterConfig,
    MeanSTDFilterConfig,
    ShannonFilterConfig,
)
from careamics.config.support.supported_filters import (
    SupportedPatchFilters,
)

from .mask_filter import MaskFilter
from .max_filter import MaxPatchFilter
from .mean_std_filter import MeanStdPatchFilter
from .shannon_filter import ShannonPatchFilter

PatchFilter = Union[
    MaskFilter,
    MaxPatchFilter,
    MeanStdPatchFilter,
    ShannonPatchFilter,
]


def create_patch_filter(filter_config: FilterConfig) -> PatchFilter:
    """Factory function to create patch filter instances based on the filter name.

    Parameters
    ----------
    filter_config : FilterConfig
        Pydantic config of the filter to be created.

    Returns
    -------
    PatchFilter
        Instance of the requested patch filter.
    """
    if filter_config.name == SupportedPatchFilters.MAX:
        assert isinstance(filter_config, MaxFilterConfig)
        return MaxPatchFilter(
            threshold=filter_config.threshold, coverage=filter_config.coverage
        )
    elif filter_config.name == SupportedPatchFilters.MEANSTD:
        assert isinstance(filter_config, MeanSTDFilterConfig)
        return MeanStdPatchFilter(
            mean_threshold=filter_config.mean_threshold,
            std_threshold=filter_config.std_threshold,
        )
    elif filter_config.name == SupportedPatchFilters.SHANNON:
        assert isinstance(filter_config, ShannonFilterConfig)
        return ShannonPatchFilter(threshold=filter_config.threshold)
    else:
        raise ValueError(f"Unknown filter name: {filter_config}")
