from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, TypeVar, Union, overload

from numpy.typing import NDArray
from typing_extensions import ParamSpec

from careamics.dataset_ng.patch_extractor import PatchExtractor
from careamics.file_io.read import ReadFunc

from .image_stack import (
    ImageStack,
    InMemoryImageStack,
    ManagedLazyImageStack,
    ZarrImageStack,
)
from .image_stack_loader import (
    ImageStackLoader,
)

P = ParamSpec("P")
GenericImageStack = TypeVar("GenericImageStack", bound=ImageStack, covariant=True)


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
    data_config: DataConfig
        The data configuration, `data_config.data_type` should have the value "array",
        and `data_config.axes` should describe the axes of every array in the `source`.

    Returns
    -------
    PatchExtractor
    """
    image_stacks = [
        InMemoryImageStack.from_array(data=array, axes=axes) for array in source
    ]
    return PatchExtractor(image_stacks)


@overload
def create_tiff_extractor(
    source: Sequence[Path], axes: str, in_mem: Literal[True]
) -> PatchExtractor[InMemoryImageStack]: ...


@overload
def create_tiff_extractor(
    source: Sequence[Path], axes: str, in_mem: Literal[False]
) -> PatchExtractor[ManagedLazyImageStack]: ...


# TIFF case
def create_tiff_extractor(
    source: Sequence[Path], axes: str, in_mem: bool
) -> Union[PatchExtractor[InMemoryImageStack], PatchExtractor[ManagedLazyImageStack]]:
    """
    Create a patch extractor from a sequence of files that match our supported types.

    Supported file types include TIFF.

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
    image_stacks: Union[Sequence[InMemoryImageStack], Sequence[ManagedLazyImageStack]]
    if in_mem:
        image_stacks = [
            InMemoryImageStack.from_tiff(path=path, axes=axes) for path in source
        ]
        return PatchExtractor(image_stacks)
    else:
        image_stacks = [
            ManagedLazyImageStack.from_tiff(path=path, axes=axes) for path in source
        ]
        return PatchExtractor(image_stacks)


# ZARR case
def create_ome_zarr_extractor(
    source: Sequence[Path],
    axes: str,
) -> PatchExtractor[ZarrImageStack]:
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


# Custom ImageStackLoader case
def create_custom_image_stack_extractor(
    source: Any,
    axes: str,
    image_stack_loader: ImageStackLoader[P, GenericImageStack],
    *args: P.args,
    **kwargs: P.kwargs,
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
    image_stacks = image_stack_loader(source, axes, *args, **kwargs)
    return PatchExtractor(image_stacks)
