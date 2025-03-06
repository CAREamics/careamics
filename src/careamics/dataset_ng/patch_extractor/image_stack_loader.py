from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, Literal, Optional, ParamSpec, Union, overload

from numpy.typing import NDArray

from careamics.config.support import SupportedData
from careamics.file_io.read import ReadFunc

from .image_stack import ImageStack, InMemoryImageStack, ZarrImageStack
from .patch_extractor import PatchExtractor

P = ParamSpec("P")
ImageStackLoader = Callable[P, Sequence[ImageStack]]


def from_arrays(source: Sequence[NDArray], axes: str) -> list[InMemoryImageStack]:
    return [InMemoryImageStack.from_array(data=array, axes=axes) for array in source]


def from_tiff_files(source: Sequence[Path], axes: str) -> list[InMemoryImageStack]:
    return [InMemoryImageStack.from_tiff(path=path, axes=axes) for path in source]


def from_custom_file_type(
    source: Sequence[Path], axes: str, read_func: ReadFunc, read_kwargs: dict[str, Any]
) -> list[InMemoryImageStack]:
    return [
        InMemoryImageStack.from_custom_file_type(
            path=path,
            axes=axes,
            read_func=read_func,
            **read_kwargs,
        )
        for path in source
    ]


def from_ome_zarr_files(source: Sequence[Path]) -> list[ZarrImageStack]:
    return [ZarrImageStack.from_ome_zarr(path) for path in source]


def get_image_stack_loader(
    data_type: Union[SupportedData, str],
    image_stack_loader: Optional[ImageStackLoader] = None,
) -> ImageStackLoader:
    if data_type == SupportedData.ARRAY:
        return from_arrays
    elif data_type == SupportedData.TIFF:
        return from_tiff_files
    elif data_type == "zarr":  # temp for testing until zarr is added to SupportedData
        return from_ome_zarr_files
    elif data_type == SupportedData.CUSTOM:
        if image_stack_loader is None:
            return from_custom_file_type
        else:
            return image_stack_loader
    else:
        raise ValueError


@overload
def create_patch_extractor(
    data_type: Literal[SupportedData.ARRAY],
    source: Sequence[NDArray],
    image_stack_loader: Literal[None] = None,
    *,
    axes: str = "",
) -> PatchExtractor: ...


@overload
def create_patch_extractor(
    data_type: Literal[SupportedData.TIFF],
    source: Sequence[Path],
    image_stack_loader: Literal[None] = None,
    *,
    axes: str = "",
) -> PatchExtractor: ...


@overload
def create_patch_extractor(
    data_type: Literal["zarr"],
    source: Sequence[Path],
    image_stack_loader: Literal[None] = None,
) -> PatchExtractor: ...


@overload
def create_patch_extractor(
    data_type: Literal[SupportedData.CUSTOM],
    source: Sequence[Path],
    image_stack_loader: Literal[None] = None,
    *,
    axes: str,
    read_func: ReadFunc,
    read_kwargs: dict[str, Any],
) -> PatchExtractor: ...


@overload
def create_patch_extractor(
    data_type: Literal[SupportedData.CUSTOM],
    source: Any,
    image_stack_loader: ImageStackLoader,
    **kwargs,
) -> PatchExtractor: ...


def create_patch_extractor(
    data_type: Union[SupportedData, str],
    source: Any,
    image_stack_loader: Optional[ImageStackLoader] = None,
    **kwargs,
) -> PatchExtractor:
    loader = get_image_stack_loader(data_type, image_stack_loader)
    image_stacks = loader(source, **kwargs)
    return PatchExtractor(image_stacks)
