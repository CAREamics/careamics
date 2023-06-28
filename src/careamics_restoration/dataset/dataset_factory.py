from functools import partial
from typing import Callable, Union

from torch.utils.data import Dataset

from ..config import ConfigStageEnum, Configuration
from ..config.stage import Stage
from ..manipulation import create_patch_transform
from .dataset import PatchDataset
from .dataset_utils import (
    extract_patches_predict,
    extract_patches_random,
    extract_patches_sequential,
    list_input_source_tiff,
)

# TODO Joran: removing factories for now


def create_tiling_function(stage: Stage) -> Union[None, Callable]:
    """Creates the tiling function depending on the provided strategy.

    Parameters
    ----------
    config : dict.

    Returns
    -------
    Callable
    """
    # TODO add proper option selection !
    if stage.data.extraction_strategy == "predict" and stage.data.patch_size is None:
        return None
    elif stage.data.extraction_strategy == "predict":
        return partial(
            extract_patches_predict,
            overlaps=stage.overlap,
        )
    elif stage.data.extraction_strategy == "sequential":
        return partial(
            extract_patches_sequential,
        )
    elif stage.data.extraction_strategy == "random":
        return partial(
            extract_patches_random,
        )
    # TODO Igor: move partial to dataset class
    return None


def create_dataset(config: Configuration, stage: ConfigStageEnum) -> Dataset:
    """Builds a dataset based on the dataset_params.

    Parameters
    ----------
    config : Dict
        Config file dictionary
    """
    stage_config = config.get_stage_config(stage)  # getattr(config, stage)

    # TODO clear description of what all these funcs/params mean
    dataset = PatchDataset(
        data_path=stage_config.data.path,
        ext=stage_config.data.ext,
        axes=stage_config.data.axes,
        num_files=stage_config.data.num_files,  # TODO this can be None (see config)
        data_reader=list_input_source_tiff,
        patch_size=stage_config.data.patch_size,
        patch_generator=create_tiling_function(stage_config),
        patch_level_transform=create_patch_transform(config)
        if stage != "prediction"
        else None,  # TODO Igor: move all funcs that return callables to dataset class
        # TODO Igor: separate dataset class for different datatypes, tiff, zarr
    )

    return dataset
