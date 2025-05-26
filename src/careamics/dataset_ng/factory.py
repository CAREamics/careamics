from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, TypeVar, Union, overload

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeIs

from careamics.config import DataConfig, InferenceConfig
from careamics.config.support import SupportedData
from careamics.dataset_ng.patch_extractor import PatchExtractor
from careamics.dataset_ng.patch_extractor.image_stack import (
    GenericImageStack,
    ImageStack,
    InMemoryImageStack,
    ZarrImageStack,
)
from careamics.dataset_ng.patch_extractor.patch_extractor_factory import (
    create_array_extractor,
    create_custom_file_extractor,
    create_custom_image_stack_extractor,
    create_ome_zarr_extractor,
    create_tiff_extractor,
)
from careamics.file_io.read import ReadFunc

from .dataset import CareamicsDataset, Mode

SourceType = Union[Sequence[NDArray[Any]], Sequence[Path], Sequence[ImageStack]]


# array
@overload
def create_dataset(
    config: Union[DataConfig, InferenceConfig],
    mode: Mode,
    inputs: Sequence[NDArray],
    targets: Optional[Sequence[NDArray]],
    in_memory: bool = True,
) -> CareamicsDataset[InMemoryImageStack]: ...


# tiff or zarr
@overload
def create_dataset(
    config: Union[DataConfig, InferenceConfig],
    mode: Mode,
    inputs: Sequence[Path],
    targets: Optional[Sequence[Path]],
    in_memory: bool,
) -> CareamicsDataset[ImageStack]: ...


# custom file
@overload
def create_dataset(
    config: Union[DataConfig, InferenceConfig],
    mode: Mode,
    inputs: Sequence[Path],
    targets: Optional[Sequence[Path]],
    in_memory: bool,
    read_func: Optional[ReadFunc],
    read_kwargs: Optional[dict[str, Any]],
) -> CareamicsDataset[ImageStack]: ...


# custom image stack
@overload
def create_dataset(
    config: Union[DataConfig, InferenceConfig],
    mode: Mode,
    inputs: Sequence[GenericImageStack],
    targets: Optional[Sequence[GenericImageStack]],
    in_memory: bool,
) -> CareamicsDataset[GenericImageStack]: ...


def create_dataset(
    config: Union[DataConfig, InferenceConfig],
    mode: Mode,
    inputs: SourceType,
    targets: Optional[SourceType],
    in_memory: bool = True,
    read_func: Optional[ReadFunc] = None,
    read_kwargs: Optional[dict[str, Any]] = None,
) -> CareamicsDataset:
    if _is_source_item_type(inputs, np.ndarray):
        if config.data_type != SupportedData.ARRAY:
            raise ValueError(
                "Data type 'array' selected in config, is not compatible with "
                f"`{type(inputs[0])}` input type."
            )
        if (targets is not None) and (not _is_source_item_type(targets, np.ndarray)):
            raise TypeError(
                "Input and target types do not match, inputs are arrays but found "
                f"target type '{type(targets[0])}'."
            )
        return create_array_dataset(config, mode, inputs, targets)
    elif _is_source_item_type(inputs, Path):
        if (targets is not None) and (not _is_source_item_type(targets, Path)):
            raise TypeError(
                "Input and target types do not match, inputs are paths but found "
                f"target type '{type(targets[0])}'."
            )
        return _create_from_paths(
            config, mode, inputs, targets, in_memory, read_func, read_kwargs
        )
    # assume remaining input type option is Sequence[ImageStack]
    else:
        if config.data_type != SupportedData.CUSTOM:
            raise ValueError(
                f"Data type `{type(inputs[0])}` is not compatible with selected "
                f"data type '{config.data_type}'. Please use 'custom' when using "
                "CAREamics with a custom `ImageStack` class."
            )
        if (targets is not None) and (
            _is_source_item_type(targets, np.ndarray)
            or _is_source_item_type(targets, Path)
        ):
            raise TypeError(
                f"Input and target types do not match, inputs are '{type(inputs[0])}' "
                f"but found target type '{type(targets[0])}'."
            )
        return create_custom_image_stack_dataset(config, mode, inputs, targets)


def _create_from_paths(
    config: Union[DataConfig, InferenceConfig],
    mode: Mode,
    inputs: Sequence[Path],
    targets: Optional[Sequence[Path]],
    in_memory: bool = True,
    read_func: Optional[ReadFunc] = None,
    read_kwargs: Optional[dict[str, Any]] = None,
):
    if config.data_type == SupportedData.TIFF:
        return create_tiff_dataset(config, mode, inputs, targets)
        # TODO: Lazy tiff

    # TODO: add ZARR to supported datatypes once old dataset is deprecated
    # elif config.data_type == SupportedData.ZARR
    elif config.data_type == SupportedData.CUSTOM:
        if read_func is None:
            raise ValueError(
                f"Data type '{SupportedData.CUSTOM.value}' has been selected in config "
                "and input type is `Path` but no `read_func` has been provided."
            )
        if read_kwargs is None:
            read_kwargs = {}
        return create_custom_file_dataset(
            config, mode, inputs, targets, read_func=read_func, read_kwargs=read_kwargs
        )
    else:
        raise ValueError(
            f"Data type '{config.data_type}' is not compatible with `Path` input type."
        )


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
    config : DataConfig or InferenceConfig
        The data configuration.
    mode : Mode
        Whether to create the dataset in "training", "validation" or "predicting" mode.
    inputs : Any
        The input sources to the dataset.
    targets : Any, optional
        The target sources to the dataset.

    Returns
    -------
    CareamicsDataset[InMemoryImageStack]
        A CAREamicsDataset
    """
    input_extractor = create_array_extractor(source=inputs, axes=config.axes)
    target_extractor: Optional[PatchExtractor[InMemoryImageStack]]
    if targets is not None:
        target_extractor = create_array_extractor(source=targets, axes=config.axes)
    else:
        target_extractor = None
    return CareamicsDataset(config, mode, input_extractor, target_extractor)


def create_tiff_dataset(
    config: Union[DataConfig, InferenceConfig],
    mode: Mode,
    inputs: Sequence[Path],
    targets: Optional[Sequence[Path]],
) -> CareamicsDataset[InMemoryImageStack]:
    """
    Create a CAREamicsDataset from tiff files that will be all loaded into memory.

    Parameters
    ----------
    config : DataConfig or InferenceConfig
        The data configuration.
    mode : Mode
        Whether to create the dataset in "training", "validation" or "predicting" mode.
    inputs : Any
        The input sources to the dataset.
    targets : Any, optional
        The target sources to the dataset.

    Returns
    -------
    CareamicsDataset[InMemoryImageStack]
        A CAREamicsDataset
    """
    input_extractor = create_tiff_extractor(
        source=inputs,
        axes=config.axes,
    )
    target_extractor: Optional[PatchExtractor[InMemoryImageStack]]
    if targets is not None:
        target_extractor = create_tiff_extractor(source=targets, axes=config.axes)
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
    """
    Create a dataset from OME ZARR files.

    Parameters
    ----------
    config : DataConfig or InferenceConfig
        The data configuration.
    mode : Mode
        Whether to create the dataset in "training", "validation" or "predicting" mode.
    inputs : Any
        The input sources to the dataset.
    targets : Any, optional
        The target sources to the dataset.

    Returns
    -------
    CareamicsDataset[ZarrImageStack]
        A CAREamicsDataset
    """

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
    """
    Create a CAREamicsDataset from custom files that will be all loaded into memory.

    Parameters
    ----------
    config : DataConfig or InferenceConfig
        The data configuration.
    mode : Mode
        Whether to create the dataset in "training", "validation" or "predicting" mode.
    inputs : Any
        The input sources to the dataset.
    targets : Any, optional
        The target sources to the dataset.
    read_func : Optional[ReadFunc], optional
        A function that can that can be used to load custom data. This argument is
        ignored unless the `data_type` is "custom".
    image_stack_loader : Optional[ImageStackLoader], optional
        A function for custom image stack loading. This argument is ignored unless the
        `data_type` is "custom".

    Returns
    -------
    CareamicsDataset[InMemoryImageStack]
        A CAREamicsDataset
    """
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
    inputs: Sequence[GenericImageStack],
    targets: Optional[Sequence[GenericImageStack]],
) -> CareamicsDataset[GenericImageStack]:
    """
    Create a CAREamicsDataset from a custom `ImageStack` class.

    The custom `ImageStack` class can be loaded using the `image_stack_loader` function.

    Parameters
    ----------
    config : DataConfig or InferenceConfig
        The data configuration.
    mode : Mode
        Whether to create the dataset in "training", "validation" or "predicting" mode.
    inputs : Any
        The input sources to the dataset.
    targets : Any, optional
        The target sources to the dataset.
    image_stack_loader : ImageStackLoader
        A function for custom image stack loading. This argument is ignored unless the
        `data_type` is "custom".
    *args : Any
        Positional arguments to pass to the `image_stack_loader`.
    **kwargs : Any
        Key-word arguments to pass to the `image_stack_loader`.

    Returns
    -------
    CareamicsDataset[GenericImageStack]
        A CAREamicsDataset
    """
    input_extractor = create_custom_image_stack_extractor(inputs, config.axes)
    target_extractor: Optional[PatchExtractor[GenericImageStack]]
    if targets is not None:
        target_extractor = create_custom_image_stack_extractor(targets, config.axes)
    else:
        target_extractor = None
    return CareamicsDataset(config, mode, input_extractor, target_extractor)


# --- utils

ItemTypeVar = TypeVar("ItemTypeVar", covariant=True)


@overload
def _is_source_item_type(
    input: SourceType,
    item_type: type[NDArray],
) -> TypeIs[Sequence[NDArray[Any]]]: ...


@overload
def _is_source_item_type(
    input: SourceType,
    item_type: type[Path],
) -> TypeIs[Sequence[Path]]: ...


# function to narrow the SourceType for mypy
# can simply check the instance of the first value because SourceType already types all
# the items as the same type
def _is_source_item_type(
    input: SourceType,
    item_type: Union[type[Path], type[NDArray]],
) -> Union[TypeIs[Sequence[NDArray[Any]]], TypeIs[Sequence[Path]]]:
    return isinstance(input[0], item_type)
