from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Optional, TypeVar, Union

from numpy.typing import NDArray
from typing_extensions import ParamSpec

from careamics.config import DataConfig, InferenceConfig
from careamics.config.support import SupportedData
from careamics.file_io.read import ReadFunc

from ..patch_extractor import ImageStackLoader, PatchExtractor
from ..patch_extractor.image_stack import (
    ImageStack,
    InMemoryImageStack,
    ManagedLazyImageStack,
    ZarrImageStack,
)
from ..patch_extractor.patch_extractor_factory import (
    create_array_extractor,
    create_custom_file_extractor,
    create_custom_image_stack_extractor,
    create_ome_zarr_extractor,
    create_tiff_extractor,
)
from .dataset import CareamicsDataset, Mode

P = ParamSpec("P")
GenericImageStack = TypeVar("GenericImageStack", bound=ImageStack, covariant=True)


class DatasetType(Enum):
    ARRAY = "array"
    IN_MEM_TIFF = "in_mem_tiff"
    LAZY_TIFF = "lazy_tiff"
    IN_MEM_CUSTOM_FILE = "in_mem_custom_file"
    OME_ZARR = "ome_zarr"
    CUSTOM_IMAGE_STACK = "custom_image_stack"


# bit of a mess of if-else statements
def determine_dataset_type(
    data_type: SupportedData,
    in_memory: Optional[bool] = None,
    read_func: Optional[ReadFunc] = None,
    image_stack_loader: Optional[ImageStackLoader] = None,
) -> DatasetType:
    if data_type == SupportedData.ARRAY:
        # TODO: ignoring in_memory arg, error if False?
        return DatasetType.ARRAY
    elif data_type == SupportedData.TIFF:
        if in_memory:
            return DatasetType.IN_MEM_TIFF
        else:
            return DatasetType.LAZY_TIFF
    elif data_type == SupportedData.CUSTOM:
        if read_func is not None:
            if in_memory:
                return DatasetType.IN_MEM_CUSTOM_FILE
            else:
                raise NotImplementedError(
                    "Lazy loading has not been implemented for custom file types yet."
                )
        elif image_stack_loader is not None:
            # TODO: ignoring im_memory arg
            return DatasetType.CUSTOM_IMAGE_STACK
        else:
            raise ValueError(
                "Found `data_type='custom'` but no `read_func` or `image_stack_loader` "
                "has been provided."
            )
    # TODO: ZARR
    else:
        raise ValueError(f"Unrecognized `data_type`, '{data_type}'.")


# convenience function but should use `create_dataloader function instead`
# For lazy loading custom batch sampler also needs to be set.
def create_dataset(
    config: Union[DataConfig, InferenceConfig],
    mode: Mode,
    inputs: Any,
    targets: Any,
    in_memory: Optional[bool] = None,
    read_func: Optional[ReadFunc] = None,
    read_kwargs: Optional[dict[str, Any]] = None,
    image_stack_loader: Optional[ImageStackLoader] = None,
    image_stack_loader_kwargs: Optional[dict[str, Any]] = None,
) -> CareamicsDataset[ImageStack]:
    data_type = SupportedData(config.data_type)
    dataset_type = determine_dataset_type(
        data_type, in_memory, read_func, image_stack_loader
    )
    if dataset_type == DatasetType.ARRAY:
        return create_array_dataset(config, mode, inputs, targets)
    elif dataset_type == DatasetType.IN_MEM_TIFF:
        return create_loaded_tiff_dataset(config, mode, inputs, targets)
    elif dataset_type == DatasetType.LAZY_TIFF:
        return create_lazy_tiff_dataset(config, mode, inputs, targets)
    elif data_type == DatasetType.IN_MEM_CUSTOM_FILE:
        if read_kwargs is None:
            read_kwargs = {}
        assert read_func is not None  # should be true from `determine_dataset_type`
        return create_custom_file_dataset(
            config, mode, inputs, targets, read_func=read_func, read_kwargs=read_kwargs
        )
    elif data_type == DatasetType.CUSTOM_IMAGE_STACK:
        if image_stack_loader_kwargs is None:
            image_stack_loader_kwargs = {}
        assert image_stack_loader is not None  # should be true
        return create_custom_image_stack_dataset(
            config, mode, inputs, targets, image_stack_loader, image_stack_loader_kwargs
        )
    else:
        raise ValueError(f"Unrecognized dataset type, {dataset_type}.")


def create_array_dataset(
    config: Union[DataConfig, InferenceConfig],
    mode: Mode,
    inputs: Sequence[NDArray[Any]],
    targets: Optional[Sequence[NDArray[Any]]],
) -> CareamicsDataset[InMemoryImageStack]:
    """
    Create a CAREamicsDataset from array data.

    Parameters
    ----------
    config : Union[DataConfig, InferenceConfig]
        _description_
    mode : Mode
        _description_
    inputs : Sequence[NDArray[Any]]
        _description_
    targets : Optional[Sequence[NDArray[Any]]]
        _description_

    Returns
    -------
    CareamicsDataset[InMemoryImageStack]
        _description_
    """
    input_extractor = create_array_extractor(source=inputs, axes=config.axes)
    target_extractor: Optional[PatchExtractor[InMemoryImageStack]]
    if targets is not None:
        target_extractor = create_array_extractor(source=targets, axes=config.axes)
    else:
        target_extractor = None
    dataset = CareamicsDataset(config, mode, input_extractor, target_extractor)
    return dataset


def create_loaded_tiff_dataset(
    config: Union[DataConfig, InferenceConfig],
    mode: Mode,
    inputs: Sequence[Path],
    targets: Optional[Sequence[Path]],
) -> CareamicsDataset[InMemoryImageStack]:
    input_extractor = create_tiff_extractor(
        source=inputs, axes=config.axes, in_mem=True
    )
    target_extractor: Optional[PatchExtractor[InMemoryImageStack]]
    if targets is not None:
        target_extractor = create_tiff_extractor(
            source=targets, axes=config.axes, in_mem=True
        )
    else:
        target_extractor = None
    dataset = CareamicsDataset(config, mode, input_extractor, target_extractor)
    return dataset


def create_lazy_tiff_dataset(
    config: Union[DataConfig, InferenceConfig],
    mode: Mode,
    inputs: Sequence[Path],
    targets: Optional[Sequence[Path]],
) -> CareamicsDataset[ManagedLazyImageStack]:
    input_extractor = create_tiff_extractor(
        source=inputs, axes=config.axes, in_mem=False
    )
    target_extractor: Optional[PatchExtractor[ManagedLazyImageStack]]
    if targets is not None:
        target_extractor = create_tiff_extractor(
            source=targets, axes=config.axes, in_mem=False
        )
    else:
        target_extractor = None
    dataset = CareamicsDataset(config, mode, input_extractor, target_extractor)
    return dataset


def create_ome_zarr_dataset(
    config: Union[DataConfig, InferenceConfig],
    mode: Mode,
    inputs: Sequence[Path],
    targets: Optional[Sequence[Path]],
) -> CareamicsDataset[ZarrImageStack]:

    input_extractor = create_ome_zarr_extractor(source=inputs, axes=config.axes)
    target_extractor: Optional[PatchExtractor[ZarrImageStack]]
    if targets is not None:
        target_extractor = create_ome_zarr_extractor(source=targets, axes=config.axes)
    else:
        target_extractor = None
    dataset = CareamicsDataset(config, mode, input_extractor, target_extractor)
    return dataset


def create_custom_file_dataset(
    config: Union[DataConfig, InferenceConfig],
    mode: Mode,
    inputs: Sequence[Path],
    targets: Optional[Sequence[Path]],
    *,
    read_func: ReadFunc,
    read_kwargs: dict[str, Any],
) -> CareamicsDataset[InMemoryImageStack]:
    input_extractor = create_custom_file_extractor(
        source=inputs, axes=config.axes, read_func=read_func, read_kwargs=read_kwargs
    )
    target_extractor: Optional[PatchExtractor[InMemoryImageStack]]
    if targets is not None:
        target_extractor = create_custom_file_extractor(
            source=targets,
            axes=config.axes,
            read_func=read_func,
            read_kwargs=read_kwargs,
        )
    else:
        target_extractor = None
    dataset = CareamicsDataset(config, mode, input_extractor, target_extractor)
    return dataset


def create_custom_image_stack_dataset(
    config: Union[DataConfig, InferenceConfig],
    mode: Mode,
    inputs: Any,
    targets: Optional[Any],
    image_stack_loader: ImageStackLoader[P, GenericImageStack],
    *args: P.args,
    **kwargs: P.kwargs,
) -> CareamicsDataset[GenericImageStack]:
    input_extractor = create_custom_image_stack_extractor(
        inputs,
        config.axes,
        image_stack_loader,
        *args,
        **kwargs,
    )
    target_extractor: Optional[PatchExtractor[GenericImageStack]]
    if targets is not None:
        target_extractor = create_custom_image_stack_extractor(
            targets,
            config.axes,
            image_stack_loader,
            *args,
            **kwargs,
        )
    else:
        target_extractor = None
    dataset = CareamicsDataset(config, mode, input_extractor, target_extractor)
    return dataset
