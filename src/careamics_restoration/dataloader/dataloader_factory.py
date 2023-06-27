from functools import partial
from typing import Callable, Union

from torch.utils.data import Dataset

from ..config import ConfigStageEnum, Configuration
from ..config.stage import Stage
from ..manipulation import create_patch_transform
from .dataloader import PatchDataset
from .dataloader_utils import (
    extract_patches_predict,
    extract_patches_random,
    extract_patches_sequential,
    list_input_source_tiff,
)


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

    return None


def create_dataset(config: Configuration, stage: ConfigStageEnum) -> Dataset:
    """Builds a dataset based on the dataset_params.

    Parameters
    ----------
    config : Dict
        Config file dictionary
    """
    # TODO rewrite this ugly bullshit. registry,etc!
    # TODO data reader getattr
    stage_config = config.get_stage_config(stage)  # getattr(config, stage)

    # TODO clear description of what all these funcs/params mean
    # TODO patch transform should be properly imported from somewhere?
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
        else None,
    )
    # TODO getatr manipulate
    # try:
    #     dataset_class = getattr(dataloader, dataset_name)
    # except ImportError:
    #     raise ImportError('Dataset not found')
    return dataset
