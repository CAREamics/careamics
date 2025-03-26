from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, TypeVar, Union

from numpy.typing import NDArray
from typing_extensions import ParamSpec

from careamics.config import DataConfig, InferenceConfig
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


def create_array_dataset(
    config: Union[DataConfig, InferenceConfig],
    mode: Mode,
    inputs: Sequence[NDArray[Any]],
    targets: Optional[Sequence[NDArray[Any]]],
) -> CareamicsDataset[InMemoryImageStack]:
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
