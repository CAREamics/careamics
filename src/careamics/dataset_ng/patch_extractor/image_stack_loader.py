from collections.abc import Sequence
from pathlib import Path
from typing import (
    Any,
    Optional,
    ParamSpec,
    Protocol,
    Union,
    overload,
)

from numpy.typing import NDArray

from careamics.config import DataConfig
from careamics.config.support import SupportedData
from careamics.file_io.read import ReadFunc

from .image_stack import ImageStack, InMemoryImageStack, ZarrImageStack
from .patch_extractor import PatchExtractor

P = ParamSpec("P")


class ImageStackLoader(Protocol[P]):

    def __call__(
        self, source: Any, data_config: DataConfig, *args: P.args, **kwargs: P.kwargs
    ) -> Sequence[ImageStack]: ...


def from_arrays(
    source: Sequence[NDArray], data_config: DataConfig, *args, **kwargs
) -> list[InMemoryImageStack]:
    axes = data_config.axes
    return [InMemoryImageStack.from_array(data=array, axes=axes) for array in source]


def from_tiff_files(
    source: Sequence[Path], data_config: DataConfig, *args, **kwargs
) -> list[InMemoryImageStack]:
    axes = data_config.axes
    return [InMemoryImageStack.from_tiff(path=path, axes=axes) for path in source]


def from_custom_file_type(
    source: Sequence[Path],
    data_config: DataConfig,
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
    source: Sequence[Path], data_config: DataConfig, *args, **kwargs
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


@overload
def create_patch_extractor(
    source: Sequence[NDArray],
    data_config: DataConfig,
) -> PatchExtractor:
    """
    Create a patch extractor from a sequence of numpy arrays.

    Parameters
    ----------
    source: sequence of numpy.ndarray
        The source arrays of the data.
    data_config: DataConfig
        The data configuration, `data_config.data_type` should have the value "array",
        and `data_config.axes` should describe the axes of every array in the `source`.

    Returns
    -------
    PatchExtractor
    """


@overload
def create_patch_extractor(
    source: Sequence[Path],
    data_config: DataConfig,
) -> PatchExtractor:
    """
    Create a patch extractor from a sequence of files that match our supported types.

    Supported file types include TIFF and ZARR.

    If the files are ZARR files they must follow the OME standard. If you have ZARR
    files that do not follow the OME standard, see documentation on how to create
    a custom `image_stack_loader`. (TODO: Add link).

    Parameters
    ----------
    source: sequence of Path
        The source files for the data.
    data_config: DataConfig
        The data configuration, `data_config.data_type` should have the value "tiff" or
        "zarr", and `data_config.axes` should describe the axes of every image in the
        `source`.

    Returns
    -------
    PatchExtractor
    """


@overload
def create_patch_extractor(
    source: Any,
    data_config: DataConfig,
    *,
    read_func: ReadFunc,
    read_kwargs: dict[str, Any],
) -> PatchExtractor:
    """
    Create a patch extractor from a sequence of files of a custom type.

    Parameters
    ----------
    source: sequence of Path
        The source files for the data.
    data_config: DataConfig
        The data configuration, `data_config.data_type` should have the value "custom".
    read_func : ReadFunc
        A function to read the custom file type, see the `ReadFunc` protocol.
    read_kwargs : dict of {str: Any}
        Kwargs that will be passed to the custom `read_func`.

    Returns
    -------
    PatchExtractor
    """


@overload
def create_patch_extractor(
    source: Any,
    data_config: DataConfig,
    image_stack_loader: ImageStackLoader[P],
    *args: P.args,
    **kwargs: P.kwargs,
) -> PatchExtractor:
    """
    Create a patch extractor using a custom `ImageStackLoader`.

    The custom image stack loader must follow the `ImageStackLoader` protocol, i.e.
    it must have the following function signature:
    ```
    def image_loader_example(
        source: Any, data_config: DataConfig, *args, **kwargs
    ) -> Sequence[ImageStack]:
    ```

    Parameters
    ----------
    source: sequence of Path
        The source files for the data.
    data_config: DataConfig
        The data configuration, `data_config.data_type` should have the value "custom".
    image_stack_loader: ImageStackLoader
        A custom image stack loader callable.
    *args: Any
        Positional arguments that will be passed to the custom image stack loader.
    **kwargs: Any
        Keyword arguments that will be passed to the custom image stack loader.

    Returns
    -------
    PatchExtractor
    """


def create_patch_extractor(
    source: Any,
    data_config: DataConfig,
    image_stack_loader: Optional[ImageStackLoader[P]] = None,
    *args: P.args,
    **kwargs: P.kwargs,
) -> PatchExtractor:
    loader = get_image_stack_loader(data_config.data_type, image_stack_loader)
    image_stacks = loader(source, data_config, *args, **kwargs)
    return PatchExtractor(image_stacks)
