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
    SupportedCoordinateFilters,
    SupportedPatchFilters,
)
from careamics.dataset_ng.image_stack import GenericImageStack
from careamics.dataset_ng.patch_extractor import PatchExtractor

from .mask_filter import MaskFilter
from .max_filter import MaxPatchFilter
from .mean_std_filter import MeanStdPatchFilter
from .shannon_filter import ShannonPatchFilter

PatchFilter = Union[
    MaxPatchFilter,
    MeanStdPatchFilter,
    ShannonPatchFilter,
]


CoordFilter = Union[MaskFilter]


def create_mask_filter(
    filter_config: MaskFilterConfig, mask: PatchExtractor[GenericImageStack]
) -> CoordFilter:
    """Factory function to create coordinate filter instances based on the filter name.

    Parameters
    ----------
    filter_model : FilterModel
        Pydantic model of the filter to be created.
    mask : PatchExtractor[GenericImageStack]
        Mask extractor to be used for the mask filter.

    Returns
    -------
    CoordFilter
        Instance of the mask patch filter.
    """
    if filter_config.name == SupportedCoordinateFilters.MASK:
        return MaskFilter(
            mask_extractor=mask, **filter_config.model_dump(exclude={"name"})
        )
    else:
        raise ValueError(f"Unknown filter name: {filter_config}")


def create_patch_filter(filter_config: FilterConfig) -> PatchFilter:
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
    if filter_config.name == SupportedPatchFilters.MAX:
        assert isinstance(filter_config, MaxFilterConfig)
        return MaxPatchFilter(**filter_config.model_dump(exclude={"name"}))
    elif filter_config.name == SupportedPatchFilters.MEANSTD:
        assert isinstance(filter_config, MeanSTDFilterConfig)
        return MeanStdPatchFilter(**filter_config.model_dump(exclude={"name"}))
    elif filter_config.name == SupportedPatchFilters.SHANNON:
        assert isinstance(filter_config, ShannonFilterConfig)
        return ShannonPatchFilter(**filter_config.model_dump(exclude={"name"}))
    else:
        raise ValueError(f"Unknown filter name: {filter_config}")
