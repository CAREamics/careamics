"""Protocol and types for loading image stacks from data sources."""

from collections.abc import Sequence
from typing import Any, Protocol

from typing_extensions import ParamSpec

from ..image_stack import GenericImageStack

P = ParamSpec("P")


class ImageStackLoader(Protocol[P, GenericImageStack]):
    """Protocol to define how ImageStacks are loaded from a source.

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
    """

    def __call__(
        self, source: Any, axes: str, *args: P.args, **kwargs: P.kwargs
    ) -> Sequence[GenericImageStack]:
        """Load `ImageStacks` from a source.

        Parameters
        ----------
        source : Any
            Data source (paths, store, etc.).
        axes : str
            Axis order (e.g. "SYX", "SCZYX").
        *args : P.args
            Additional positional arguments for loading.
        **kwargs : P.kwargs
            Additional keyword arguments for loading.

        Returns
        -------
        Sequence[GenericImageStack]
            The loaded ImageStacks.
        """
        ...
