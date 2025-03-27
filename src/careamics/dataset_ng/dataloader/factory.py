import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, TypeVar, Union

import numpy as np
from torch.utils.data import DataLoader

from careamics.config import DataConfig, InferenceConfig
from careamics.config.support import SupportedData
from careamics.dataset.dataset_utils import get_files_size
from careamics.file_io.read import ReadFunc
from careamics.utils import get_ram_size

from ..batch_sampler import FifoBatchSampler
from ..dataset import CareamicsDataset, Mode
from ..dataset.factory import (
    DatasetType,
    create_array_dataset,
    create_custom_file_dataset,
    create_custom_image_stack_dataset,
    create_lazy_tiff_dataset,
    determine_dataset_type,
)
from ..patch_extractor import ImageStackLoader
from ..patch_extractor.image_stack import ImageStack, ManagedLazyImageStack

GenericImageStack = TypeVar("GenericImageStack", bound=ImageStack, covariant=True)


def calc_max_files_loadable(files: list[Path], max_mem_usage: float = 0.8) -> int:
    """
    Calculate the maximum number of files that will fit into memory.

    Parameters
    ----------
    files : list of Path
        Paths to the files wished to be loaded.
    max_mem_usage : float, default=0.8
        The maximum percentage of available memory that can be used to load the files.

    Returns
    -------
    int
        The maximum number of files that could fit into the specified percentage of
        available memory.
    """
    available = get_ram_size()
    file_sizes = [f.stat().st_size / 1024**2 for f in files]
    file_sizes = sorted(file_sizes)[::-1]  # largest to smallest
    cumsum = np.cumsum([f.stat().st_size / 1024**2 for f in files])
    cumsum_ratio = cumsum / available
    n = np.where(cumsum_ratio < max_mem_usage)[0][-1]
    return n


def create_dataloader(
    config: Union[DataConfig, InferenceConfig],
    mode: Mode,
    inputs: Any,
    targets: Optional[Any] = None,
    batch_size: int = 1,
    max_mem_use: float = 0.8,
    read_func: Optional[ReadFunc] = None,
    read_kwargs: Optional[dict[str, Any]] = None,
    image_stack_loader: Optional[ImageStackLoader] = None,
    image_stack_loader_kwargs: Optional[dict[str, Any]] = None,
    **dataloader_kwargs: Any,
) -> DataLoader[CareamicsDataset[ImageStack]]:
    """
    Create a dataloader for the CAREamics dataset.

    Parameters
    ----------
    config : DataConfig or InferenceConfig
        The data configuration.
    mode : Mode
        Whether to create the dataset in "training", "validation" or "predicting" mode.
    inputs : Any
        The input sources to the dataset.
    batch_size : int, default=1
        The batch size.
    max_mem_usage : float, default=0.8
        The maximum percentage of available memory that can be used to load the files.
    read_func : ReadFunc, optional
        A function that can that can be used to load custom data. This argument is
        ignored unless the `data_type` in the `config` is "custom".
    read_kwargs : dict of {str, Any}, optional
        Additional key-word arguments to pass to the `read_func`.
    image_stack_loader : ImageStackLoader, optional
        A function for custom image stack loading. This argument is ignored unless the
        `data_type` in the `config` is "custom".
    image_stack_loader_kwargs : {str, Any}, optional
        Additional key-word arguments to pass to the `image_stack_loader`.
    **dataloader_kwargs : Any
        Additional key-word arguments to pass to the dataloader. See the PyTorch docs.

    Returns
    -------
    DataLoader[CareamicsDataset[ImageStack]]
        A dataloader for the CAREamicsDataset.

    Raises
    ------
    ValueError
        For an unrecognized `data_type` in the `config`.
    """
    # calculate total size
    all_files = list(inputs)
    if targets is not None:
        all_files.extend(targets)
    total_file_size = get_files_size(all_files)
    in_memory: bool = total_file_size < get_ram_size() * max_mem_use

    data_type = SupportedData(config.data_type)

    dataset_type = determine_dataset_type(
        data_type, in_memory, read_func, image_stack_loader
    )
    if dataset_type == DatasetType.ARRAY:
        dataset = create_array_dataset(config, mode, inputs, targets)
        return DataLoader(dataset, batch_size, **dataloader_kwargs)
    elif dataset_type == DatasetType.IN_MEM_TIFF:
        dataset = create_lazy_tiff_dataset(config, mode, inputs, targets)
        return DataLoader(dataset, batch_size, **dataloader_kwargs)
    elif dataset_type == DatasetType.LAZY_TIFF:
        max_files = calc_max_files_loadable(all_files, max_mem_use)
        # make max files even because fifo_manager is shared between input and target
        if (max_files % 2 != 0) and (max_files != 1):
            max_files = max_files - 1
        return create_lazy_tiff_dataloader(
            config,
            mode,
            inputs,
            targets,
            batch_size,
            max_files_loaded=max_files,
            **dataloader_kwargs,
        )
    elif data_type == DatasetType.IN_MEM_CUSTOM_FILE:
        if read_kwargs is None:
            read_kwargs = {}
        assert read_func is not None  # should be true from `determine_dataset_type`
        dataset = create_custom_file_dataset(
            config, mode, inputs, targets, read_func=read_func, read_kwargs=read_kwargs
        )
        return DataLoader(dataset, batch_size, **dataloader_kwargs)
    elif data_type == DatasetType.CUSTOM_IMAGE_STACK:
        if image_stack_loader_kwargs is None:
            image_stack_loader_kwargs = {}
        assert image_stack_loader is not None  # should be true
        dataset = create_custom_image_stack_dataset(
            config,
            mode,
            inputs,
            targets,
            image_stack_loader=image_stack_loader,
            **image_stack_loader_kwargs,
        )
        return DataLoader(dataset, batch_size, **dataloader_kwargs)
    else:
        raise ValueError(f"Unrecognized dataset type, {dataset_type}.")


def create_lazy_tiff_dataloader(
    config: Union[DataConfig, InferenceConfig],
    mode: Mode,
    inputs: Sequence[Path],
    targets: Optional[Sequence[Path]] = None,
    batch_size: int = 1,
    max_files_loaded: int = 2,
    **dataloader_kwargs: Any,
) -> DataLoader[CareamicsDataset[ManagedLazyImageStack]]:
    """
    Create a dataloader for lazily loading TIFF files.

    Sets the batch sampler to be the `FifoBatchSampler`.

    Parameters
    ----------
    config : DataConfig or InferenceConfig
        The data configuration.
    mode : Mode
        Whether to create the dataset in "training", "validation" or "predicting" mode.
    inputs : Any
        The input sources to the dataset.
    batch_size : int, default=1
        The batch size.
    max_files_loaded : int, default=2
        The maximum number of files that should have their data loaded at one time.

    Returns
    -------
    DataLoader[CareamicsDataset[ManagedLazyImageStack]]
        A dataloader for lazily loading TIFF files, using the CAREamicsDataset and
        FifoBatchSampler.

    Raises
    ------
    ValueError
        If the `data_type` in the `config` is not "tiff".
    """
    data_type = SupportedData(config.data_type)
    if data_type != SupportedData.TIFF:
        raise ValueError(
            "To create a TIFF dataloader data type in config must be "
            f"'tiff' found {DataConfig.data_type}."
        )

    num_workers = dataloader_kwargs.get("num_workers", 0)
    if num_workers > 0:
        warnings.warn(
            "Using lazy loading and found `num_workers>0`, but lazy loading has "
            "not been optimized for multiprocessing. We suggest setting "
            "`num_workers=0`. Note: Lazy loading has been selected because the "
            "data has been calculated to not fit into memory.",
            category=Warning,
            stacklevel=1,
        )
    dataset = create_lazy_tiff_dataset(config, mode, inputs, targets)
    batch_sampler_kwargs = {}
    if "shuffle" in dataloader_kwargs:
        batch_sampler_kwargs["shuffle"] = dataloader_kwargs["shuffle"]
        del dataloader_kwargs["shuffle"]
    if "drop_last" in dataloader_kwargs:
        batch_sampler_kwargs["drop_last"] = dataloader_kwargs["drop_last"]
        del dataloader_kwargs["drop_last"]

    batch_sampler = FifoBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        max_files_loaded=max_files_loaded,
        # TODO: set random seed arg?
        **batch_sampler_kwargs,
    )
    data_loader = DataLoader(
        dataset=dataset, batch_sampler=batch_sampler, **dataloader_kwargs
    )
    return data_loader
