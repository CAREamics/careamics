import inspect
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch

from . import dataloader, pixel_manipulation
from .augment import augment_single
from .config import ConfigValidator
from .dataloader import (
    PatchDataset,
    open_input_source_tiff,
)
from .losses import n2v_loss
from .models import UNet


def _get_params_from_config(
    func: Union[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler],
    user_params: Dict,
) -> Dict:
    """Returns the parameters of the optimizer or lr_scheduler.

    Parameters
    ----------
    func : Union[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]
        optimizer or lr_scheduler class object
    user_params : Dict
        The parameters from user-provided config file

    Returns
    -------
    Dict
        The parameters of the optimizer or lr_scheduler
    """
    # TODO not restrict to optim and lr_scheduler?

    # Get the list of all default parameters
    default_params = list(inspect.signature(func).parameters.keys())
    # Retrieve provided parameters
    params_to_be_used = set(user_params.keys()) & set(default_params)
    return {key: user_params[key] for key in params_to_be_used}


def _get_params_from_module(func: Callable, user_params: List) -> list:
    """Returns the parameters of the function.

    Parameters
    ----------
    func : Callable
        The name of the optimizer or lr_scheduler
    user_params : Dict
        The parameters from user-provided config file

    Returns
    -------
    list
        The parameters of the optimizer or lr_scheduler
    """
    # Get the list of all default parameters
    default_params = list(inspect.signature(func).parameters.keys())
    # Retrieve provided parameters
    params_to_be_used = list(set(user_params) & set(default_params))
    return params_to_be_used


# TODO add get from config general function!!
def get_from_config(
    config: ConfigValidator,
    key: str,
    default: Optional[Union[str, int, float, bool]] = None,
) -> Union[str, int, float, bool, None]:
    """Returns the value of the key from the config file.

    Parameters
    ----------
    config : Dict
        The config file
    key : str
        The key to be retrieved
    default : Optional[Union[str, int, float, bool]], optional
        The default value, by default None

    Returns
    -------
    Union[str, int, float, bool]
        The value of the key
    """
    if key in config:
        return config[key]
    else:
        return default


def create_patch_transform(config: ConfigValidator) -> Callable:
    """Creates the patch transform function with optional augmentation
    Parameters
    ----------
    config : dict.

    Returns
    -------
    Callable
    """
    return partial(
        getattr(
            pixel_manipulation, f"{config.algorithm.pixel_manipulation}_manipulate"
        ),
        num_pixels=config.algorithm.num_masked_pixels,
        # TODO add augmentation selection
        augmentations=None,
    )


def create_dataset(config: ConfigValidator, stage: str) -> torch.utils.data.Dataset:
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

    if stage_config.data.ext == "tif":
        # TODO put this into a separate function?
        patch_generation_func = partial(
            getattr(
                dataloader,
                f"extract_patches_{stage_config.data.extraction_strategy}",
            ),
            overlap=config.prediction.overlap,
        )
        # TODO clear description of what all these funcs/params mean
        # TODO patch transform should be properly imported from somewhere?
        dataset = PatchDataset(
            data_path=stage_config.data.path,
            num_files=stage_config.data.num_files,
            data_reader=open_input_source_tiff,
            patch_size=stage_config.data.patch_size,
            patch_generator=patch_generation_func,
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


def create_model(config: Dict) -> torch.nn.Module:
    """Builds a model based on the model_name or load a checkpoint.

    Parameters
    ----------
    config : Dict
        Config file dictionary
    """
    # TODO rewrite this ugly bullshit. registry,etc!
    model_name = config.algorithm.model
    load_checkpoint = config.algorithm.checkpoint
    # TODO fix import
    # try:
    #     model_class = getattr(deconoising, model_name)
    # except ImportError:
    #     raise ImportError('Model not found')

    if model_name == "UNet":
        model = UNet(config.algorithm.conv_mult)
    if load_checkpoint:
        # TODO add logging message
        model.load_state_dict(torch.load(load_checkpoint))
    return model


def create_optimizer(
    optimizer_type: str, optimizer_params: Dict
) -> Tuple[torch.optim.Optimizer, Dict]:
    """Builds a model based on the model_name or load a checkpoint.

    _extended_summary_

    Parameters
    ----------
    model_name : _type_
        _description_
    """
    # assert inspect.get
    optimizer = getattr(torch.optim, optimizer_type)
    # Get the list of all possible parameters of the optimizer
    params = _get_params(optimizer, optimizer_params)
    return optimizer, params


def create_lr_scheduler(
    scheduler_type: str, scheduler_params: Dict
) -> Tuple[torch.optim.lr_scheduler._LRScheduler, Dict]:
    """Builds a model based on the model_name or load a checkpoint.

    _extended_summary_

    Parameters
    ----------
    model_name : _type_
        _description_
    """
    scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)
    params = _get_params_from_config(scheduler, scheduler_params)
    return scheduler, params


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


def create_grad_scaler(scaler_type):
    """Builds a model based on the model_name or load a checkpoint.

    _extended_summary_

    Parameters
    ----------
    model_name : _type_
        _description_
    """
    return getattr(torch.cuda.amp, scaler_type)
