"""ImageStack implementation for in-memory data."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Self, Union

import numpy as np
from numpy.typing import DTypeLike, NDArray

from careamics.file_io.read import ReadFunc, read_tiff
from careamics.utils.reshape_array import reshape_array

from .image_utils.image_stack_utils import channel_slice, pad_patch


class InMemoryImageStack:
    """
    ImageStack with data already loaded in memory.

    Parameters
    ----------
    source : Path or "array"
        Origin of the data, either a path to a file or the string `"array"` for numpy
        arrays.
    data : numpy.ndarray
        Array with axes SC(Z)YX.
    original_axes : str or None, optional
        Axis in original order.
    original_data_shape : tuple of int or None, optional
        Shape in original axis order.
    """

    def __init__(
        self,
        source: Union[Path, Literal["array"]],
        data: NDArray,
        original_axes: str | None = None,
        original_data_shape: tuple[int, ...] | None = None,
    ):
        """Constructor.

        Parameters
        ----------
        source : Path or "array"
            Origin of the data, either a path to a file or the string `"array"` for
            numpy arrays.
        data : numpy.ndarray
            Array with axes SC(Z)YX.
        original_axes : str or None, optional
            Axis in original order.
        original_data_shape : tuple of int or None, optional
            Shape in original axis order.
        """
        self.source: Union[str, Path, Literal["array"]] = source
        # data expected to be in SC(Z)YX shape, reason to use from_array constructor
        self._data: NDArray = data
        self.data_shape: Sequence[int] = self._data.shape
        self.data_dtype: DTypeLike = self._data.dtype
        self._original_axes: str = original_axes if original_axes is not None else ""
        self._original_data_shape: tuple[int, ...] = (
            original_data_shape if original_data_shape is not None else ()
        )

    def extract_patch(
        self,
        sample_idx: int,
        channels: Sequence[int] | None,  # `channels = None` to select all channels
        coords: Sequence[int],
        patch_size: Sequence[int],
    ) -> NDArray:
        """Extract a patch for a given sample and channels within the image stack.

        Parameters
        ----------
        sample_idx : int
            Sample index.
        channels : sequence of int or None
            Channel indices to extract. If `None`, all channels will be extracted.
        coords : sequence of int
            Spatial coordinates of the top-left corner of the patch.
        patch_size : sequence of int
            Size of the patch in each spatial dimension.

        Returns
        -------
        numpy.ndarray
            A patch of the image data from a particular sample with dimensions C(Z)YX.
        """
        if (coord_dims := len(coords)) != (patch_dims := len(patch_size)):
            raise ValueError(
                "Patch coordinates and patch size must have the same dimensions but "
                f"found {coord_dims} ({coords}) and {patch_dims} ({patch_size})."
            )

        # check that channels are within bounds
        if channels is not None:
            max_channel = self.data_shape[1] - 1  # channel is second dimension
            for ch in channels:
                if ch > max_channel:
                    raise ValueError(
                        f"Channel index {ch} is out of bounds for data with "
                        f"{self.data_shape[1]} channels. Check the provided `channels` "
                        f"parameter in the configuration for erroneous channel "
                        f"indices."
                    )

        # TODO: test for 2D or 3D?

        patch_data = self._data[
            (
                sample_idx,  # type: ignore
                # use channel slice so that channel dimension is kept
                channel_slice(channels),  # type: ignore
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

    @property
    def original_data_shape(self) -> tuple[int, ...]:
        """Original shape of the data.

        Returns
        -------
        tuple of int
            Shape in original axis order.
        """
        return self._original_data_shape

    @property
    def original_axes(self) -> str:
        """Original axes of the data.

        Returns
        -------
        str
            Axis order string.
        """
        return self._original_axes

    @classmethod
    def from_array(cls, data: NDArray, axes: str) -> Self:
        """Construct an in-memory stack from an array.

        Parameters
        ----------
        data : numpy.ndarray
            Array (any axis order).
        axes : str
            Axis order of the data.

        Returns
        -------
        Self
            In-memory stack.
        """
        return cls(
            source="array",
            data=reshape_array(data, axes),
            original_axes=axes,
            original_data_shape=data.shape,
        )

    @classmethod
    def from_tiff(cls, path: Path, axes: str) -> Self:
        """Build an in-memory stack from a TIFF file.

        Parameters
        ----------
        path : Path
            Path to TIFF file.
        axes : str
            Axis order of the data.

        Returns
        -------
        Self
            In-memory stack.
        """
        data = read_tiff(path)
        return cls(
            source=path,
            data=reshape_array(data, axes),
            original_axes=axes,
            original_data_shape=data.shape,
        )

    @classmethod
    def from_custom_file_type(
        cls, path: Path, axes: str, read_func: ReadFunc, **read_kwargs: Any
    ) -> Self:
        """Build an in-memory stack from a custom file type.

        Parameters
        ----------
        path : Path
            Path to file.
        axes : str
            Axis order of the data.
        read_func : ReadFunc
            Function to read the file.
        **read_kwargs : Any
            Additional keyword arguments passed to `read_func`.

        Returns
        -------
        Self
            In-memory stack.
        """
        data = read_func(path, **read_kwargs)
        return cls(
            source=path,
            data=reshape_array(data, axes),
            original_axes=axes,
            original_data_shape=data.shape,
        )
