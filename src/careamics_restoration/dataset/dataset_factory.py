from functools import partial
from typing import Callable, List, Optional, Union

from ..config import ConfigStageEnum, Configuration
from ..config.training import ExtractionStrategies
from ..manipulation import create_masking_transform
from .dataset import PatchDataset
from .dataset_utils import (
    extract_patches_predict,
    extract_patches_random,
    extract_patches_sequential,
    list_input_source_tiff,
)


# TODO this needs to be refactored and rewritten
def create_tiling_function(
    strategy: str,
    patch_size: Optional[List[int]] = None,
    overlaps: Optional[List[int]] = None,
) -> Union[None, Callable]:
    if strategy == ExtractionStrategies.TILED and patch_size is None:
        return None
    elif strategy == ExtractionStrategies.TILED:
        return partial(
            extract_patches_predict,
            overlaps=overlaps,
        )
    elif strategy == ExtractionStrategies.SEQUENTIAL:
        return partial(
            extract_patches_sequential,
        )
    elif strategy == ExtractionStrategies.RANDOM:
        return partial(
            extract_patches_random,
        )
    return None


# TODO this needs to be refactored and rewritten
def create_dataset(config: Configuration, stage: ConfigStageEnum) -> PatchDataset:
    """Builds a dataset based on the dataset_params.

    Parameters
    ----------
    config : Dict
        Config file dictionary
    """
    if stage == ConfigStageEnum.TRAINING:
        if config.training is None:
            raise ValueError("Training configuration is not defined.")

        dataset = PatchDataset(
            data_path=config.data.training_path,  # TODO this can be None
            ext=config.data.data_format,
            axes=config.data.axes,
            data_reader=list_input_source_tiff,
            patch_size=config.training.patch_size,
            patch_generator=create_tiling_function(config.training.extraction_strategy),
            patch_level_transform=create_masking_transform(config),
        )
    elif stage == ConfigStageEnum.VALIDATION:
        if config.training is None:
            raise ValueError("Training configuration is not defined.")

        dataset = PatchDataset(
            data_path=config.data.validation_path,  # TODO this can be None
            ext=config.data.data_format,
            axes=config.data.axes,
            data_reader=list_input_source_tiff,
            patch_size=config.training.patch_size,
            patch_generator=create_tiling_function(config.training.extraction_strategy),
            patch_level_transform=create_masking_transform(config),
        )
    elif stage == ConfigStageEnum.PREDICTION:
        if config.prediction is None:
            raise ValueError("Prediction configuration is not defined.")

        dataset = PatchDataset(
            data_path=config.data.prediction_path,
            ext=config.data.data_format,
            axes=config.data.axes,
            data_reader=list_input_source_tiff,
            patch_size=config.prediction.tile_shape,  # TODO this can be None
            patch_generator=create_tiling_function(
                ExtractionStrategies.TILED,
                config.prediction.tile_shape,
                config.prediction.overlaps,
            ),
            patch_level_transform=None,
        )

    return dataset
