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
    from ...batch_sampler import FifoImageStackManager

logger = getLogger(name=__name__)


class ManagedLazyImageStack:

    def __init__(
        self,
        path: Path,
        axes: str,
        data_shape: tuple[int, ...],
        data_dtype: DTypeLike,
        read_func: ReadFunc,
        read_kwargs: Optional[dict[str, Any]] = None,
        on_load: Optional[Callable[["ManagedLazyImageStack"], None]] = None,
        on_close: Optional[Callable[["ManagedLazyImageStack"], None]] = None,
    ):
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
        return self.path

    # helps with readability in the FifoImageStackManager
    @property
    def is_loaded(self):
        return self._data is not None

    def extract_patch(
        self, sample_idx: int, coords: Sequence[int], patch_size: Sequence[int]
    ) -> NDArray:
        if self._data is None:
            self.load()
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

    # loosely coupling with FifoImageStackManager
    def register_manager(self, manager: "FifoImageStackManager"):
        self._on_load = manager.notify_load
        self._on_close = manager.notify_close
        manager.register_image_stack(self)

    def load(self):
        if self._on_load is not None:
            self._on_load(self)
        data = self.read_func(self.path, **self.read_kwargs)
        data = reshape_array(data, self._original_axes)
        self._data = data
        logger.info(f"Loaded file '{self.path}'.")

    def deallocate(self):
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
