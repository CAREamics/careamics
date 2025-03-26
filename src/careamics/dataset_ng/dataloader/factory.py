import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Optional, TypeVar, Union

import numpy as np
from torch.utils.data import DataLoader

from careamics.config import DataConfig, InferenceConfig
from careamics.config.support import SupportedData
from careamics.dataset.dataset_utils import get_files_size
from careamics.file_io.read import ReadFunc
from careamics.utils import get_ram_size

from ..batch_sampler import FifoImageStackManager, GroupedBatchSampler
from ..dataset import Mode
from ..dataset.factory import (
    create_array_dataset,
    create_custom_file_dataset,
    create_custom_image_stack_dataset,
    create_lazy_tiff_dataset,
    create_loaded_tiff_dataset,
)
from ..patch_extractor import ImageStackLoader
from ..patch_extractor.image_stack import (
    ImageStack,
)

GenericImageStack = TypeVar("GenericImageStack", bound=ImageStack, covariant=True)


def calc_max_files_loadable(files: list[Path], max_mem_usage: float = 0.8) -> int:
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
):
    if config.data_type == SupportedData.ARRAY:
        dataset = create_array_dataset(config, mode, inputs, targets)
        return DataLoader(dataset, batch_size, **dataloader_kwargs)
    elif config.data_type == SupportedData.TIFF:
        dataloader = create_tiff_dataloader(
            config,
            mode,
            inputs,
            targets,
            batch_size,
            max_mem_use=max_mem_use,
            **dataloader_kwargs,
        )
        return dataloader
    elif config.data_type == SupportedData.CUSTOM:
        if read_func is not None:
            if read_kwargs is None:
                read_kwargs = {}
            return create_custom_file_dataset(
                config,
                mode,
                inputs,
                targets,
                read_func=read_func,
                read_kwargs=read_kwargs,
            )
        elif image_stack_loader is not None:
            if image_stack_loader_kwargs is None:
                image_stack_loader_kwargs = {}
            return create_custom_image_stack_dataset(
                config,
                mode,
                inputs,
                targets,
                image_stack_loader=image_stack_loader,
                **image_stack_loader_kwargs,
            )
        else:
            raise ValueError(
                "Found `data_type='custom'` but no `read_func` or `image_stack_loader` "
                "has been provided."
            )
    # TODO: Zarr
    else:
        raise ValueError(f"Unrecognized `data_type`, '{config.data_type}'.")


def create_tiff_dataloader(
    config: Union[DataConfig, InferenceConfig],
    mode: Mode,
    inputs: Sequence[Path],
    targets: Optional[Sequence[Path]] = None,
    batch_size: int = 1,
    max_mem_use: float = 0.8,
    **dataloader_kwargs: Any,
) -> DataLoader:
    data_type = SupportedData(config.data_type)
    if data_type != SupportedData.TIFF:
        raise ValueError(
            "To create a TIFF dataloader data type in config must be "
            f"'tiff' found {DataConfig.data_type}."
        )

    # calculate total size
    all_files = list(inputs)
    if targets is not None:
        all_files.extend(targets)
    total_file_size = get_files_size(all_files)
    in_memory: Literal[True, False] = total_file_size < get_ram_size() * max_mem_use

    if in_memory:
        dataset = create_loaded_tiff_dataset(config, mode, inputs, targets)
        data_loader = DataLoader(dataset=dataset, **dataloader_kwargs)
    else:
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
        batch_sampler_kwargs: dict[str, Any] = {}
        if "shuffle" in dataloader_kwargs:
            batch_sampler_kwargs["shuffle"] = dataloader_kwargs["shuffle"]
            del dataloader_kwargs["shuffle"]
        if "drop_last" in dataloader_kwargs:
            batch_sampler_kwargs["drop_last"] = dataloader_kwargs["drop_last"]
            del dataloader_kwargs["drop_last"]

        max_files = calc_max_files_loadable(all_files, max_mem_use)
        # make max files even because fifo_manager is shared between input and target
        if (max_files % 2 != 0) and (max_files != 1):
            max_files = max_files - 1
        batch_sampler = GroupedBatchSampler(
            dataset=dataset, batch_size=batch_size, **batch_sampler_kwargs
        )

        # TODO: put this somewhere it is accessible, maybe in the dataset
        # make fifo manager and register file
        fifo_manager = FifoImageStackManager(max_files_loaded=max_files)
        fifo_manager.register_image_stacks(dataset.input_extractor.image_stacks)
        if dataset.target_extractor is not None:
            fifo_manager.register_image_stacks(dataset.target_extractor.image_stacks)

        data_loader = DataLoader(
            dataset=dataset, batch_sampler=batch_sampler, **dataloader_kwargs
        )
    return data_loader
