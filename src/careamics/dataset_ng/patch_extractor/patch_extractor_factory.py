from collections.abc import Sequence
from pathlib import Path
from typing import Union

from numpy.typing import NDArray

from careamics.config import DataConfig
from careamics.config.inference_model import InferenceConfig
from careamics.config.support import SupportedData
from careamics.dataset_ng.patch_extractor import (
    PatchExtractor,
    PatchExtractorConstructor,
)


def get_patch_extractor_constructor(
    data_config: Union[DataConfig, InferenceConfig],
) -> PatchExtractorConstructor:
    if data_config.data_type == SupportedData.ARRAY:
        return PatchExtractor.from_arrays
    elif data_config.data_type == SupportedData.TIFF:
        return PatchExtractor.from_tiff_files
    elif data_config.data_type == SupportedData.CUSTOM:
        return PatchExtractor.from_custom_file_type
    else:
        raise ValueError(f"Data type {data_config.data_type} is not supported.")


def create_patch_extractor(
    data_config: Union[DataConfig, InferenceConfig],
    data: Union[Sequence[NDArray], Sequence[Path]],
    **kwargs,
) -> PatchExtractor:
    # get correct constructor
    constructor = get_patch_extractor_constructor(data_config)

    # build key word args
    constructor_kwargs = {"axes": data_config.axes, **kwargs}

    # --- data extractor
    patch_extractor: PatchExtractor = constructor(source=data, **constructor_kwargs)

    return patch_extractor
