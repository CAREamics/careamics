from collections.abc import Sequence
from pathlib import Path
from typing import Any

from numpy.typing import NDArray
from typing_extensions import ParamSpec

from careamics.dataset_ng.patch_extractor import PatchExtractor
from careamics.file_io.read import ReadFunc

from .image_stack import (
    GenericImageStack,
    InMemoryImageStack,
    ZarrImageStack,
)

P = ParamSpec("P")


# Array case
def create_array_extractor(
    source: Sequence[NDArray[Any]], axes: str
) -> PatchExtractor[InMemoryImageStack]:
    """
    Create a patch extractor from a sequence of numpy arrays.

    Parameters
    ----------
    source: sequence of numpy.ndarray
        The source arrays of the data.
    axes: str
        The original axes of the data, must be a subset of "STCZYX".

    Returns
    -------
    PatchExtractor
    """
    image_stacks = [
        InMemoryImageStack.from_array(data=array, axes=axes) for array in source
    ]
    return PatchExtractor(image_stacks)


# TIFF case
def create_tiff_extractor(
    source: Sequence[Path], axes: str
) -> PatchExtractor[InMemoryImageStack]:
    """
    Create a patch extractor from a sequence of TIFF files.

    Parameters
    ----------
    source: sequence of Path
        The source files for the data.
    axes: str
        The original axes of the data, must be a subset of "STCZYX".

    Returns
    -------
    PatchExtractor
    """
    image_stacks = [
        InMemoryImageStack.from_tiff(path=path, axes=axes) for path in source
    ]
    return PatchExtractor(image_stacks)


# ZARR case
def create_ome_zarr_extractor(
    source: Sequence[Path],
    axes: str,
) -> PatchExtractor[ZarrImageStack]:
    """
    Create a patch extractor from a sequence of OME ZARR files.

    If you have ZARR files that do not follow the OME standard, see documentation on
    how to create a custom `image_stack_loader`. (TODO: Add link).

    Parameters
    ----------
    source: sequence of Path
        The source files for the data.
    axes: str
        The original axes of the data, must be a subset of "STCZYX".

    Returns
    -------
    PatchExtractor
    """
    # NOTE: axes is unused here, in from_ome_zarr the axes are automatically retrieved
    image_stacks = [ZarrImageStack.from_ome_zarr(path) for path in source]
    return PatchExtractor(image_stacks)


# Custom file type case (loaded into memory)
def create_custom_file_extractor(
    source: Sequence[Path],
    axes: str,
    *,
    read_func: ReadFunc,
    read_kwargs: dict[str, Any],
) -> PatchExtractor[InMemoryImageStack]:
    """
    Create a patch extractor from a sequence of files of a custom type.

    Parameters
    ----------
    source: sequence of Path
        The source files for the data.
    axes: str
        The original axes of the data, must be a subset of "STCZYX".
    read_func : ReadFunc
        A function to read the custom file type, see the `ReadFunc` protocol.
    read_kwargs : dict of {str: Any}
        Kwargs that will be passed to the custom `read_func`.

    Returns
    -------
    PatchExtractor
    """
    # TODO: lazy loading custom files
    image_stacks = [
        InMemoryImageStack.from_custom_file_type(
            path=path,
            axes=axes,
            read_func=read_func,
            **read_kwargs,
        )
        for path in source
    ]

    return PatchExtractor(image_stacks)


# Custom ImageStack case
def create_custom_image_stack_extractor(
    source: Sequence[GenericImageStack],
    axes: str,
) -> PatchExtractor[GenericImageStack]:
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
    axes: str
        The original axes of the data, must be a subset of "STCZYX".
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
    return PatchExtractor(source)
