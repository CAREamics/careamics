from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

from numpy.typing import NDArray

from careamics.config import DataConfig
from careamics.config.support import SupportedData
from careamics.dataset_ng.patch_extractor import (
    PatchExtractor,
    PatchExtractorConstructor,
)


def get_patch_extractor_constructor(
    data_config: DataConfig,
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
    data_config: DataConfig,
    data: Union[Sequence[NDArray], Sequence[Path]],
    target_data: Optional[Union[Sequence[NDArray], Sequence[Path]]] = None,
    **kwargs,
) -> tuple[PatchExtractor, Optional[PatchExtractor]]:
    # get correct constructor
    constructor = get_patch_extractor_constructor(data_config)

    # build key word args
    constructor_kwargs = {"axes": data_config.axes, **kwargs}

    # --- data extractor
    patch_extractor: PatchExtractor = constructor(source=data, **constructor_kwargs)
    # --- optional target extractor
    if target_data is not None:
        target_patch_extractor: PatchExtractor = constructor(
            source=target_data, **constructor_kwargs
        )

        return patch_extractor, target_patch_extractor

    return patch_extractor, None
