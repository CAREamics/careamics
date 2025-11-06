"""Factories for coordinate and patch filters."""

from typing import Union

from careamics.config.data.patch_filter import (
    FilterModel,
    MaskFilterModel,
    MaxFilterModel,
    MeanSTDFilterModel,
    ShannonFilterModel,
)
from careamics.config.support.supported_filters import (
    SupportedCoordinateFilters,
    SupportedPatchFilters,
)
from careamics.dataset_ng.patch_extractor import GenericImageStack, PatchExtractor

from .mask_filter import MaskCoordFilter
from .max_filter import MaxPatchFilter
from .mean_std_filter import MeanStdPatchFilter
from .shannon_filter import ShannonPatchFilter

PatchFilter = Union[
    MaxPatchFilter,
    MeanStdPatchFilter,
    ShannonPatchFilter,
]


CoordFilter = Union[MaskCoordFilter]


def create_coord_filter(
    filter_model: FilterModel, mask: PatchExtractor[GenericImageStack]
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
    if filter_model.name == SupportedCoordinateFilters.MASK:
        assert isinstance(filter_model, MaskFilterModel)
        return MaskCoordFilter(
            mask_extractor=mask,
            coverage=filter_model.coverage,
            p=filter_model.p,
            seed=filter_model.seed,
        )
    else:
        raise ValueError(f"Unknown filter name: {filter_model}")


def create_patch_filter(filter_model: FilterModel) -> PatchFilter:
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
        assert isinstance(filter_model, MaxFilterModel)
        return MaxPatchFilter(
            threshold=filter_model.threshold, p=filter_model.p, seed=filter_model.seed
        )
    elif filter_model.name == SupportedPatchFilters.MEANSTD:
        assert isinstance(filter_model, MeanSTDFilterModel)
        return MeanStdPatchFilter(
            mean_threshold=filter_model.mean_threshold,
            std_threshold=filter_model.std_threshold,
            p=filter_model.p,
            seed=filter_model.seed,
        )
    elif filter_model.name == SupportedPatchFilters.SHANNON:
        assert isinstance(filter_model, ShannonFilterModel)
        return ShannonPatchFilter(
            threshold=filter_model.threshold, p=filter_model.p, seed=filter_model.seed
        )
    else:
        raise ValueError(f"Unknown filter name: {filter_model}")
