from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

import numpy as np
import tifffile
from numpy.typing import DTypeLike, NDArray

from careamics.dataset.dataset_utils import reshape_array
from careamics.file_io.read import ReadFunc, read_tiff

from .utils import pad_patch, reshaped_array_shape


class FileImageStack:
    """
    An ImageStack implementation for data that is coming from a file.

    The data will not be loaded until the `load` method is called. The `close` method
    can be used to remove the internal reference to the data.
    """

    def __init__(
        self,
        source: Path,
        axes: str,
        data_shape: tuple[int, ...],
        data_dtype: DTypeLike,
        read_func: ReadFunc,
        read_kwargs: dict[str, Any] | Any = None,
    ):
        self.source = source
        self.axes = axes
        self.data_shape = data_shape
        self.data_dtype = data_dtype
        self.read_func = read_func
        self.read_kwargs = read_kwargs
        self._data: NDArray | None = None

    def extract_patch(
        self, sample_idx: int, coords: Sequence[int], patch_size: Sequence[int]
    ) -> NDArray:
        if self._data is None:
            raise ValueError(
                "Cannot extract patch because data has not been loaded from "
                f"'{self.source}', the `load` method must be called first."
            )

        if (coord_dims := len(coords)) != (patch_dims := len(patch_size)):
            raise ValueError(
                "Patch coordinates and patch size must have the same dimensions but "
                f"found {coord_dims} and {patch_dims}."
            )

        patch_data = self._data[
            (
                sample_idx,  # type: ignore
                ...,  # type: ignore
                *[
                    slice(
                        np.clip(c, 0, self.data_shape[2 + i]),
                        np.clip(c + ps, 0, self.data_shape[2 + i]),
                    )
                    for i, (c, ps) in enumerate(zip(coords, patch_size, strict=False))
                ],  # type: ignore
            )  # type: ignore
        ]
        patch = pad_patch(coords, patch_size, self.data_shape, patch_data)

        return patch

    def load(self):
        """Load the data stored in a file."""
        data = self.read_func(self.source)
        self._data = reshape_array(data, self.axes)

    # TODO: maybe this should be called something else
    def close(self):
        """Remove the internal reference to the data to clear up memory."""
        # will get cleaned up by the garbage collector since there is no longer a ref
        self._data = None

    @property
    def is_loaded(self):
        return self._data is not None

    @classmethod
    def from_tiff(
        cls,
        path: Path,
        axes: str,
    ) -> Self:
        """
        Construct the `ImageStack` from a TIFF file.

        Parameters
        ----------
        path : Path
            Path to the TIFF file.
        axes : str
            The original axes of the data, must be a subset of STCZYX.

        Returns
        -------
        Self
            The `ImageStack` with the underlying data being from a TIFF file.
        """
        # TODO: think this is correct but need more examples to test
        file = tifffile.TiffFile(path)
        data_shape = reshaped_array_shape(axes, file.series[0].shape)
        dtype = file.series[0].dtype
        return cls(
            source=path,
            axes=axes,
            data_shape=data_shape,
            data_dtype=dtype,
            read_func=read_tiff,
        )
