from collections.abc import Sequence
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import tifffile
from numpy.typing import DTypeLike, NDArray
from typing_extensions import Self

from careamics.dataset.dataset_utils import reshape_array
from careamics.file_io.read import ReadFunc, read_tiff

from ._utils import reshaped_array_shape

if TYPE_CHECKING:
    pass

# TODO: better name
LazyCallback = Callable[["ManagedLazyImageStack"], None]

logger = getLogger(name=__name__)


class ManagedLazyImageStack:
    """
    Implements the `ImageStack` Protocol. The data for this `ImageStack` is stored in
    a file. The data is not loaded until the first time `extract_patch` is called, at
    which point the data is stored as an attribute within the class. If `deallocate` is
    called the reference to the data is dropped so python's garbage collector can clean
    it up.

    This `ImageStack` also has a callback system so that it can be managed by an
    external class. It can take two optional callback functions, `on_load` and
    `on_close`, these can be used to perform additional logic when the ImageStack loads
    its data and when the memory of the data has been deallocated.
    """

    def __init__(
        self,
        path: Path,
        axes: str,
        data_shape: tuple[int, ...],
        data_dtype: DTypeLike,
        read_func: ReadFunc,
        read_kwargs: Optional[dict[str, Any]] = None,
        on_load: Optional[LazyCallback] = None,
        on_close: Optional[LazyCallback] = None,
    ):
        """
        An `ImageStack` for the lazy loading of data. It can also be managed through
        the callbacks `on_load` and `on_close`.

        Parameters
        ----------
        path : Path
            Path to the file that contains the image data.
        axes : str
            The original axes of the data, must be a subset of STCZYX.
        data_shape : tuple[int, ...]
            The original shape of the data.
        data_dtype : DTypeLike
            The datatype of the data.
        read_func : ReadFunc
            A function to read the data.
        read_kwargs : dict of {str, Any}, optional
            Additional key-word arguments to be passed to the `read_func`.
        on_load : Optional[LazyCallback], optional
            A callback function that will be called when the data is loaded.
        on_close : Optional[LazyCallback], optional
            A callback function that will be called when the data is deallocated.
        """
        self.path = path
        self.read_func = read_func
        self.read_kwargs = read_kwargs if read_kwargs else {}
        self._original_axes = axes
        self._original_data_shape = data_shape
        self.data_shape = reshaped_array_shape(axes, self._original_data_shape)
        self.data_dtype = data_dtype

        # Callback system
        self._on_load = on_load
        self._on_close = on_close

        self._data = None

    @property
    def source(self) -> Path:
        """The data source."""
        return self.path

    # helps with readability in the FifoImageStackManager
    @property
    def is_loaded(self):
        """If the data is currently loaded."""
        return self._data is not None

    def extract_patch(
        self, sample_idx: int, coords: Sequence[int], patch_size: Sequence[int]
    ) -> NDArray:
        """Extract a patch from the image."""
        if self._data is None:
            self.load()  # load only when extract patch is called for the first time
            assert self._data is not None
        if len(coords) != len(patch_size):
            raise ValueError("Length of coords and extent must match.")
        # TODO: test for 2D or 3D?
        return self._data[
            (
                sample_idx,  # type: ignore
                ...,  # type: ignore
                *[slice(c, c + e) for c, e in zip(coords, patch_size)],  # type: ignore
            )
        ]

    # normally on_load and on_close will be
    #   FifoImageStackManager.notify_load,
    #   FifoImageStackManager.notify_close,
    def set_callbacks(self, on_load: LazyCallback, on_close: LazyCallback):
        """
        Set the callbacks.

        Parameters
        ----------
        on_load : LazyCallback
            A callback function that will be called when the data is loaded.
        on_close : LazyCallback
            A callback function that will be called when the data is deallocated.
        """
        self._on_load = on_load
        self._on_close = on_close

    def load(self):
        """Load the data."""
        if self._on_load is not None:
            self._on_load(self)
        data = self.read_func(self.path, **self.read_kwargs)
        data = reshape_array(data, self._original_axes)
        self._data = data
        logger.info(f"Loaded file '{self.path}'.")

    def deallocate(self):
        """Remove reference to the data so the memory can be deallocated."""
        # TODO: raise error if not loaded?
        if self._on_close is not None:
            self._on_close(self)
        self._data = None  # deallocating lets python garbage collector clean up
        logger.info(f"Deallocated data from file '{self.path}'.")

    @classmethod
    def from_tiff(
        cls,
        path: Path,
        axes: str,
        on_load: Optional[Callable[["ManagedLazyImageStack"], None]] = None,
        on_close: Optional[Callable[["ManagedLazyImageStack"], None]] = None,
    ) -> Self:
        """
        Construct the `ImageStack` from a TIFF file.

        Parameters
        ----------
        path : Path
            Path to the TIFF file.
        axes : str
            The original axes of the data, must be a subset of STCZYX.
        on_load : Optional[LazyCallback], optional
            A callback function that will be called when the data is loaded.
        on_close : Optional[LazyCallback], optional
            A callback function that will be called when the data is deallocated.

        Returns
        -------
        Self
            The `ImageStack` with the underlying data being from a TIFF file.
        """
        # TODO: think this is correct but need more examples to test
        file = tifffile.TiffFile(path)
        data_shape = file.series[0].shape
        dtype = file.series[0].dtype
        return cls(
            path=path,
            axes=axes,
            data_shape=data_shape,
            data_dtype=dtype,
            read_func=read_tiff,
            on_load=on_load,
            on_close=on_close,
        )

        # TODO: from custom


if __name__ == "__main__":
    from .image_stack_protocol import ImageStack

    image_stack: ImageStack = ManagedLazyImageStack.from_tiff(
        path=Path("a/b"), axes="YX"
    )
