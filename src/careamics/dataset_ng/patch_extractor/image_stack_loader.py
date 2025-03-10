from collections.abc import Sequence
from pathlib import Path
from typing import (
    Any,
    Optional,
    Protocol,
    Union,
)

from numpy.typing import NDArray
from typing_extensions import ParamSpec

from careamics.config.support import SupportedData
from careamics.file_io.read import ReadFunc
from careamics.utils import BaseEnum

from .image_stack import ImageStack, InMemoryImageStack, ZarrImageStack

P = ParamSpec("P")


class SupportedDataDev(str, BaseEnum):
    ZARR = "zarr"


class ImageStackLoader(Protocol[P]):
    """
    Protocol to define how `ImageStacks` should be loaded.

    An `ImageStackLoader` is a callable that must take the `source` of the data as the
    first argument, and the data `axes` as the second argument.

    Additional `*args` and `**kwargs` are allowed, but they should only be used to
    determine _how_ the data is loaded, not _what_ data is loaded. The `source`
    argument has to wholly determine _what_ data is loaded, this is because,
    downstream, both an input-source and a target-source have to be specified but they
    will share `*args` and `**kwargs`.

    An `ImageStackLoader` must return a sequence of the `ImageStack` class. This could
    be a sequence of one of the existing concrete implementations, such as
    `ZarrImageStack`, or a custom user defined `ImageStack`.

    Example
    -------
    The following example demonstrates how an `ImageStackLoader` could be defined
    for loading non-OME Zarr images. Returning a list of `ZarrImageStack` instances.

    >>> from typing import TypedDict

    >>> from zarr.storage import FSStore

    >>> from careamics.config import DataConfig
    >>> from careamics.dataset_ng.patch_extractor.image_stack import ZarrImageStack

    >>> # Define a zarr source
    >>> # It encompasses multiple arguments that determine what data will be loaded
    >>> class ZarrSource(TypedDict):
    ...     store: FSStore
    ...     data_paths: Sequence[str]

    >>> def custom_image_stack_loader(
    ...     source: ZarrSource, axes: str, *args, **kwargs
    ... ) -> list[ZarrImageStack]:
    ...     image_stacks = [
    ...         ZarrImageStack(store=source["store"], data_path=data_path, axes=axes)
    ...         for data_path in source["data_paths"]
    ...     ]
    ...     return image_stacks

    TODO: show example use in the `CAREamicsDataset`

    The example above defines a `ZarrSource` dict because to determine _which_ ZARR
    images will be loaded both a ZARR store and the internal data paths need to be
    specified.
    """

    def __call__(
        self, source: Any, axes: str, *args: P.args, **kwargs: P.kwargs
    ) -> Sequence[ImageStack]: ...


def from_arrays(
    source: Sequence[NDArray], axes: str, *args, **kwargs
) -> list[InMemoryImageStack]:
    return [InMemoryImageStack.from_array(data=array, axes=axes) for array in source]


# TODO: change source to directory path? Like in current implementation
#   Advantage of having a list is the user can match input and target order themselves
def from_tiff_files(
    source: Sequence[Path], axes: str, *args, **kwargs
) -> list[InMemoryImageStack]:
    return [InMemoryImageStack.from_tiff(path=path, axes=axes) for path in source]


# TODO: change source to directory path? Like in current implementation
#   Advantage of having a list is the user can match input and target order themselves
def from_custom_file_type(
    source: Sequence[Path],
    axes: str,
    read_func: ReadFunc,
    read_kwargs: dict[str, Any],
    *args,
    **kwargs,
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


def from_ome_zarr_files(
    source: Sequence[Path], axes: str, *args, **kwargs
) -> list[ZarrImageStack]:
    # NOTE: axes is unused here, in from_ome_zarr the axes are automatically retrieved
    return [ZarrImageStack.from_ome_zarr(path) for path in source]


def get_image_stack_loader(
    data_type: Union[SupportedData, SupportedDataDev],
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
