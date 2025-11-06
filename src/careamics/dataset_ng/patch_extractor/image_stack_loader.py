from collections.abc import Sequence
from typing import Any, Protocol

from typing_extensions import ParamSpec

from careamics.utils import BaseEnum

from .image_stack import GenericImageStack

P = ParamSpec("P")


class SupportedDataDev(str, BaseEnum):
    ZARR = "zarr"


class ImageStackLoader(Protocol[P, GenericImageStack]):
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

    >>> from zarr.storage import FsspecStore

    >>> from careamics.config import DataConfig
    >>> from careamics.dataset_ng.patch_extractor.image_stack import ZarrImageStack

    >>> # Define a zarr source
    >>> # It encompasses multiple arguments that determine what data will be loaded
    >>> class ZarrSource(TypedDict):
    ...     store: FsspecStore
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
    ) -> Sequence[GenericImageStack]: ...
