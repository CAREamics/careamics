"""ImageStack implementation for Zarr-backed images."""

from collections.abc import Sequence

import numpy as np
import zarr
from numpy.typing import DTypeLike, NDArray

from careamics.utils.reshape_array import AxesTransform, get_patch_slices, reshape_patch

from .image_utils.image_stack_utils import pad_patch


class ZarrImageStack:
    """
    ImageStack backed by a zarr array.

    Parameters
    ----------
    group : zarr.Group
        Zarr group containing the array.
    data_path : str
        Path to the array within the group.
    axes : str
        Axis order (e.g. STCZYX).
    """

    def __init__(self, group: zarr.Group, data_path: str, axes: str):
        """Constructor.

        Parameters
        ----------
        group : zarr.Group
            Zarr group containing the array.
        data_path : str
            Path to the array within the group.
        axes : str
            Axis order (e.g. STCZYX).
        """
        if not isinstance(group, zarr.Group):
            raise TypeError(f"group must be a zarr.Group instance, got {type(group)}.")

        self._group = group
        self._store = str(group.store_path)
        try:
            self._array = group[data_path]
        except KeyError as e:
            raise ValueError(
                f"Did not find array at '{data_path}' in store '{self._store}'."
            ) from e

        if not isinstance(self._array, zarr.Array):
            raise TypeError(
                f"data at path '{data_path}' must be a zarr.Array instance, "
                f"got {type(self._array)}."
            )

        self._source = self._array.store_path

        # TODO: validate axes
        #   - must contain XY
        #   - must be subset of STCZYX
        self._original_axes = axes
        self._original_data_shape: tuple[int, ...] = self._array.shape
        self.data_shape = AxesTransform(
            axes, self._original_data_shape
        ).transformed_shape

        self._data_dtype = self._array.dtype
        self._chunk_size = self._array.chunks
        self._shard_size = self._array.shards

    @property
    def source(self) -> str:
        """Source URI of the zarr array.

        Local zarr URIs starts with the `file://` descriptor, and include the path to
        the zarr file and internal path to the specific array. Source URIs are used
        during prediction to disk to build destination paths.

        Returns
        -------
        str
            Source URI.
        """
        return str(self._source)

    @property
    def chunks(self) -> Sequence[int]:
        """Chunk size per dimension.

        Returns
        -------
        Sequence[int]
            Chunk size per dimension.
        """
        return self._chunk_size

    @property
    def shards(self) -> Sequence[int] | None:
        """Shard size per dimension.

        Returns
        -------
        Sequence[int] or None
            Shard size per dimension, or None.
        """
        return self._shard_size

    @property
    def data_dtype(self) -> DTypeLike:
        """Data type of the array.

        Returns
        -------
        numpy.DTypeLike
            Type of the data.
        """
        return self._data_dtype

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

    def extract_patch(
        self,
        sample_idx: int,
        channels: Sequence[int] | None,  # `channels = None` to select all channels,
        coords: Sequence[int],
        patch_size: Sequence[int],
    ) -> NDArray[np.float32]:
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
        patch_slice = get_patch_slices(
            self._original_axes,
            self._original_data_shape,
            sample_idx,
            channels,
            coords,
            patch_size,
        )

        patch_data: NDArray = self._array[patch_slice]  # type: ignore
        patch_data = reshape_patch(patch_data, self._original_axes)
        patch = pad_patch(coords, patch_size, self.data_shape, patch_data)
        return patch
