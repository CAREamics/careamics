import inspect
import logging
from functools import partial
from typing import Callable, Dict, Optional, Union

import torch

from . import dataloader
from .config import Configuration
from .utils import set_logging
from .dataloader import (
    PatchDataset,
    list_input_source_tiff,
)
from .losses.losses import n2v_loss
from .models import UNET

from .manipulation import create_patch_transform


logger = logging.getLogger(__name__)
set_logging(logger)


def create_tiling_function(stage: Dict) -> Callable:
    """Creates the tiling function depending on the provided strategy.
    Parameters
    ----------
    config : dict.

    Returns
    -------
    Callable
    """
    # TODO add proper option selection !
    if stage.data.extraction_strategy == "predict" and all(
        ps == 1 for ps in stage.data.patch_size
    ):
        return None
    elif stage.data.extraction_strategy == "predict":
        return partial(
            getattr(
                dataloader,
                f"extract_patches_{stage.data.extraction_strategy}",
            ),
            overlaps=stage.overlap,
        )
    else:
        return partial(
            getattr(
                dataloader,
                f"extract_patches_{stage.data.extraction_strategy}",
            ),
        )


def create_dataset(config: Configuration, stage: str) -> torch.utils.data.Dataset:
    """Builds a dataset based on the dataset_params.

    Parameters
    ----------
    config : Dict
        Config file dictionary
    """
    # TODO rewrite this ugly bullshit. registry,etc!
    # TODO data reader getattr
    # TODO add support for mixed filetype datasets
    stage_config = config.get_stage_config(stage)  # getattr(config, stage)

    # TODO clear description of what all these funcs/params mean
    # TODO patch transform should be properly imported from somewhere?
    dataset = PatchDataset(
        data_path=stage_config.data.path,
        ext=stage_config.data.ext,
        axes=stage_config.data.axes,
        num_files=stage_config.data.num_files,
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


def create_loss_function(config: Dict) -> Callable:
    """Builds a model based on the model_name or load a checkpoint.

    _extended_summary_

    Parameters
    ----------
    model_name : _type_
        _description_
    """
    loss_type = config.algorithm.loss
    if loss_type[0] == "n2v":
        loss_function = n2v_loss
    # TODO rewrite this ugly bullshit. registry,etc!
    # loss_func = getattr(sys.__name__, loss_type)
    # TODO test !
    return loss_function
