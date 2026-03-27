"""Factories for coordinate and patch filters."""

from typing import Union

from careamics.config.data.patch_filter import (
    FilterConfig,
    MaskFilterConfig,
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


def create_patch_filter(filter_model: FilterConfig) -> PatchFilter:
    """Factory function to create patch filter instances based on the filter name.

    Parameters
    ----------
    filter_model : FilterModel
        Pydantic model of the filter to be created.

    Returns
    -------
    PatchFilter
        Instance of the requested patch filter.
    """
    if filter_model.name == SupportedPatchFilters.MAX:
        assert isinstance(filter_model, MaxFilterConfig)
        return MaxPatchFilter(
            threshold=filter_model.threshold, coverage=filter_model.coverage
        )
    elif filter_model.name == SupportedPatchFilters.MEANSTD:
        assert isinstance(filter_model, MeanSTDFilterConfig)
        return MeanStdPatchFilter(
            mean_threshold=filter_model.mean_threshold,
            std_threshold=filter_model.std_threshold,
        )
    elif filter_model.name == SupportedPatchFilters.SHANNON:
        assert isinstance(filter_model, ShannonFilterConfig)
        return ShannonPatchFilter(threshold=filter_model.threshold)
    # TODO: add mask to enum?
    elif filter_model.name == "mask":
        assert isinstance(filter_model, MaskFilterConfig)
        return MaskFilter(coverage=filter_model.coverage)
    else:
        raise ValueError(f"Unknown filter name: {filter_model}")
