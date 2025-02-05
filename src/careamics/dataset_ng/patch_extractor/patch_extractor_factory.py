from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

from numpy.typing import NDArray

from careamics.config import GeneralDataConfig
from careamics.config.support import SupportedData
from careamics.dataset_ng.patch_extractor import (
    PatchExtractor,
    PatchExtractorConstructor,
)


def get_patch_extractor_constructor(
    data_config: GeneralDataConfig,
) -> PatchExtractorConstructor:
    if data_config.data_type == SupportedData.ARRAY:
        return PatchExtractor.from_arrays
    elif data_config.data_type == SupportedData.TIFF:
        return PatchExtractor.from_tiff_files
    elif data_config.data_type == SupportedData.CUSTOM:
        return PatchExtractor.from_custom_file_type
    else:
        raise ValueError(f"Data type {data_config.data_type} is not supported.")


def create_patch_extractors(
    data_config: GeneralDataConfig,
    train_data: Union[Sequence[NDArray], Sequence[Path]],
    val_data: Optional[Union[Sequence[NDArray], Sequence[Path]]] = None,
    train_data_target: Optional[Union[Sequence[NDArray], Sequence[Path]]] = None,
    val_data_target: Optional[Union[Sequence[NDArray], Sequence[Path]]] = None,
    **kwargs,
) -> tuple[
    PatchExtractor,
    Optional[PatchExtractor],
    Optional[PatchExtractor],
    Optional[PatchExtractor],
]:

    # get correct constructor
    constructor = get_patch_extractor_constructor(data_config)

    # build key word args
    constructor_kwargs = {"axes": data_config.axes, **kwargs}

    # --- train data extractor
    train_patch_extractor: PatchExtractor = constructor(
        source=train_data, **constructor_kwargs
    )
    # --- additional data extractors
    additional_patch_extractors: list[Union[PatchExtractor, None]] = []
    additional_data_sources = [val_data, train_data_target, val_data_target]
    for data_source in additional_data_sources:
        if data_source is not None:
            additional_patch_extractor: Optional[PatchExtractor] = constructor(
                source=data_source, **constructor_kwargs
            )
        else:
            additional_patch_extractor = None
        additional_patch_extractors.append(additional_patch_extractor)

    return (
        train_patch_extractor,
        additional_patch_extractors[0],
        additional_patch_extractors[1],
        additional_patch_extractors[2],
    )
