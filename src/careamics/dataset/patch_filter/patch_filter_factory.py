"""Factories for coordinate and patch filters."""

from typing import Union

from careamics.config.data.patch_filter import (
    MaxPatchFilterConfig,
    MeanSTDPatchFilterConfig,
    PatchFilterConfig,
    ShannonPatchFilterConfig,
)
from careamics.config.support.supported_filters import (
    SupportedPatchFilters,
)

from .mask_patch_filter import MaskPatchFilter
from .max_patch_filter import MaxPatchFilter
from .mean_std_patch_filter import MeanStdPatchFilter
from .shannon_patch_filter import ShannonPatchFilter

PatchFilter = Union[
    MaskPatchFilter,
    MaxPatchFilter,
    MeanStdPatchFilter,
    ShannonPatchFilter,
]


def create_patch_filter(filter_config: PatchFilterConfig) -> PatchFilter:
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
        assert isinstance(filter_config, MaxPatchFilterConfig)
        return MaxPatchFilter(
            threshold=filter_config.threshold, coverage=filter_config.coverage
        )
    elif filter_config.name == SupportedPatchFilters.MEANSTD:
        assert isinstance(filter_config, MeanSTDPatchFilterConfig)
        return MeanStdPatchFilter(
            mean_threshold=filter_config.mean_threshold,
            std_threshold=filter_config.std_threshold,
        )
    elif filter_config.name == SupportedPatchFilters.SHANNON:
        assert isinstance(filter_config, ShannonPatchFilterConfig)
        return ShannonPatchFilter(threshold=filter_config.threshold)
    else:
        raise ValueError(f"Unknown filter name: {filter_config}")
