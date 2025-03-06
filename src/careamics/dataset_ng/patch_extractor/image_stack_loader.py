from collections.abc import Sequence
from pathlib import Path
from typing import (
    Any,
    Optional,
    ParamSpec,
    Protocol,
    Union,
)

from numpy.typing import NDArray

from careamics.config import DataConfig
from careamics.config.support import SupportedData
from careamics.file_io.read import ReadFunc

from .image_stack import ImageStack, InMemoryImageStack, ZarrImageStack

P = ParamSpec("P")


class ImageStackLoader(Protocol[P]):

    def __call__(
        self, data_config: DataConfig, source: Any, *args: P.args, **kwargs: P.kwargs
    ) -> Sequence[ImageStack]: ...


def from_arrays(
    data_config: DataConfig, source: Sequence[NDArray], *args, **kwargs
) -> list[InMemoryImageStack]:
    axes = data_config.axes
    return [InMemoryImageStack.from_array(data=array, axes=axes) for array in source]


def from_tiff_files(
    data_config: DataConfig, source: Sequence[Path], *args, **kwargs
) -> list[InMemoryImageStack]:
    axes = data_config.axes
    return [InMemoryImageStack.from_tiff(path=path, axes=axes) for path in source]


def from_custom_file_type(
    data_config: DataConfig,
    source: Sequence[Path],
    read_func: ReadFunc,
    read_kwargs: dict[str, Any],
    *args,
    **kwargs,
) -> list[InMemoryImageStack]:
    axes = data_config.axes
    return [
        InMemoryImageStack.from_custom_file_type(
            path=path,
            axes=axes,
            read_func=read_func,
            **read_kwargs,
        )
        for path in source
    ]


def from_ome_zarr_files(
    data_config: DataConfig, source: Sequence[Path], *args, **kwargs
) -> list[ZarrImageStack]:
    return [ZarrImageStack.from_ome_zarr(path) for path in source]


def get_image_stack_loader(
    data_type: Union[SupportedData, str],  # Union with string temp for zarr
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
