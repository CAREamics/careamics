from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from numpy.typing import NDArray
from typing_extensions import ParamSpec

from careamics.config.data.ng_data_model import NGDataConfig
from careamics.config.support import SupportedData
from careamics.dataset_ng.patch_extractor import ImageStackLoader, PatchExtractor
from careamics.dataset_ng.patch_extractor.image_stack import (
    CziImageStack,
    GenericImageStack,
    ImageStack,
    InMemoryImageStack,
    ZarrImageStack,
)
from careamics.dataset_ng.patch_extractor.patch_extractor_factory import (
    create_array_extractor,
    create_custom_file_extractor,
    create_custom_image_stack_extractor,
    create_czi_extractor,
    create_ome_zarr_extractor,
    create_tiff_extractor,
)
from careamics.file_io.read import ReadFunc

from .dataset import CareamicsDataset, Mode

P = ParamSpec("P")


# Enum class used to determine which loading functions should be used
class DatasetType(Enum):
    """Labels for the dataset based on the underlying data and how it is loaded."""

    ARRAY = "array"
    IN_MEM_TIFF = "in_mem_tiff"
    LAZY_TIFF = "lazy_tiff"
    IN_MEM_CUSTOM_FILE = "in_mem_custom_file"
    OME_ZARR = "ome_zarr"
    CZI = "czi"
    CUSTOM_IMAGE_STACK = "custom_image_stack"


# bit of a mess of if-else statements
def determine_dataset_type(
    data_type: SupportedData,
    in_memory: bool,
    read_func: Optional[ReadFunc] = None,
    image_stack_loader: Optional[ImageStackLoader] = None,
) -> DatasetType:
    """Determine what the dataset type should be based on the input arguments.

    Parameters
    ----------
    data_type : SupportedData
        The underlying datatype.
    in_memory : bool
        Whether all the data should be loaded into memory. This is argument is ignored
        unless the `data_type` is "tiff" or "custom".
    read_func : ReadFunc, optional
        A function that can be used to load custom data. This argument is
        ignored unless the `data_type` is "custom".
    image_stack_loader : ImageStackLoader, optional
        A function for custom image stack loading. This argument is ignored unless the
        `data_type` is "custom".

    Returns
    -------
    DatasetType
        The Dataset type.

    Raises
    ------
    NotImplementedError
        For lazy-loading (`in_memory=False`) of a custom file type.
    ValueError
        If the `data_type` is "custom" but both `read_func` and `image_stack_loader` are
        None.
    ValueError
        If the `data_type` is unrecognized.
    """
    if data_type == SupportedData.ARRAY:
        # TODO: ignoring in_memory arg, error if False?
        return DatasetType.ARRAY
    elif data_type == SupportedData.TIFF:
        if in_memory:
            return DatasetType.IN_MEM_TIFF
        else:
            return DatasetType.LAZY_TIFF
    elif data_type == SupportedData.CZI:
        return DatasetType.CZI
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


# convenience function but should use `create_dataloader` function instead
# For lazy loading custom batch sampler also needs to be set.
def create_dataset(
    config: NGDataConfig,
    mode: Mode,
    inputs: Any,
    targets: Any,
    in_memory: bool,
    read_func: Optional[ReadFunc] = None,
    read_kwargs: Optional[dict[str, Any]] = None,
    image_stack_loader: Optional[ImageStackLoader] = None,
    image_stack_loader_kwargs: Optional[dict[str, Any]] = None,
) -> CareamicsDataset[ImageStack]:
    """
    Convenience function to create the CAREamicsDataset.

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
    in_memory : bool
        Whether all the data should be loaded into memory. This is argument is ignored
        unless the `data_type` in `config` is "tiff" or "custom".
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

    Returns
    -------
    CareamicsDataset[ImageStack]
        The CAREamicsDataset.

    Raises
    ------
    ValueError
        For an unrecognized `data_type` in the `config`.
    """
    data_type = SupportedData(config.data_type)
    dataset_type = determine_dataset_type(
        data_type, in_memory, read_func, image_stack_loader
    )
    if dataset_type == DatasetType.ARRAY:
        return create_array_dataset(config, mode, inputs, targets)
    elif dataset_type == DatasetType.IN_MEM_TIFF:
        return create_tiff_dataset(config, mode, inputs, targets)
    # TODO: Lazy tiff
    elif dataset_type == DatasetType.CZI:
        return create_czi_dataset(config, mode, inputs, targets)
    elif dataset_type == DatasetType.IN_MEM_CUSTOM_FILE:
        if read_kwargs is None:
            read_kwargs = {}
        assert read_func is not None  # should be true from `determine_dataset_type`
        return create_custom_file_dataset(
            config, mode, inputs, targets, read_func=read_func, read_kwargs=read_kwargs
        )
    elif dataset_type == DatasetType.CUSTOM_IMAGE_STACK:
        if image_stack_loader_kwargs is None:
            image_stack_loader_kwargs = {}
        assert image_stack_loader is not None  # should be true
        return create_custom_image_stack_dataset(
            config,
            mode,
            inputs,
            targets,
            image_stack_loader,
            **image_stack_loader_kwargs,
        )
    else:
        raise ValueError(f"Unrecognized dataset type, {dataset_type}.")


def create_array_dataset(
    config: NGDataConfig,
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
        A CAREamicsDataset.
    """
    input_extractor = create_array_extractor(source=inputs, axes=config.axes)
    target_extractor: Optional[PatchExtractor[InMemoryImageStack]]
    if targets is not None:
        target_extractor = create_array_extractor(source=targets, axes=config.axes)
    else:
        target_extractor = None
    return CareamicsDataset(config, mode, input_extractor, target_extractor)


def create_tiff_dataset(
    config: NGDataConfig,
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
        A CAREamicsDataset.
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


def create_czi_dataset(
    config: NGDataConfig,
    mode: Mode,
    inputs: Sequence[Path],
    targets: Optional[Sequence[Path]],
) -> CareamicsDataset[CziImageStack]:
    """
    Create a dataset from CZI files.

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
    CareamicsDataset[CziImageStack]
        A CAREamicsDataset.
    """

    input_extractor = create_czi_extractor(source=inputs, axes=config.axes)
    target_extractor: Optional[PatchExtractor[CziImageStack]]
    if targets is not None:
        target_extractor = create_czi_extractor(source=targets, axes=config.axes)
    else:
        target_extractor = None
    dataset = CareamicsDataset(config, mode, input_extractor, target_extractor)
    return dataset


def create_ome_zarr_dataset(
    config: NGDataConfig,
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
        A CAREamicsDataset.
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
    config: NGDataConfig,
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
        A CAREamicsDataset.
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
    config: NGDataConfig,
    mode: Mode,
    inputs: Any,
    targets: Optional[Any],
    image_stack_loader: ImageStackLoader[P, GenericImageStack],
    *args: P.args,
    **kwargs: P.kwargs,
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
        A CAREamicsDataset.
    """
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
