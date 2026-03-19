"""ImageStack implementation for Zarr-backed images."""

from collections.abc import Sequence

import zarr
from numpy.typing import DTypeLike, NDArray

from careamics.utils.reshape_array import AxesTransform, reshape_array

from .image_utils.image_stack_utils import channel_slice, pad_patch


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
        # original axes assumed to be any subset of STCZYX (containing YX), in any order
        # arguments must be transformed to index data in original axes order
        # to do this: loop through original axes and append correct index/slice
        #   for each case: STCZYX
        #   Note: if any axis is not present in original_axes it is skipped.

        # guard for no S and T in original axes
        if ("S" not in self._original_axes) and ("T" not in self._original_axes):
            if sample_idx not in [0, -1]:
                raise IndexError(
                    f"Sample index {sample_idx} out of bounds for S axes with size "
                    f"{self.data_shape[0]}"
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

        patch_slice: list[int | slice] = []
        for d in self._original_axes:
            if d == "S":
                patch_slice.append(self._get_S_index(sample_idx))
            elif d == "T":
                patch_slice.append(self._get_T_index(sample_idx))
            elif d == "C":
                patch_slice.append(channel_slice(channels))  # type: ignore
            elif d == "Z":
                patch_slice.append(slice(coords[0], coords[0] + patch_size[0]))
            elif d == "Y":
                y_idx = 0 if "Z" not in self._original_axes else 1
                patch_slice.append(
                    slice(coords[y_idx], coords[y_idx] + patch_size[y_idx])
                )
            elif d == "X":
                x_idx = 1 if "Z" not in self._original_axes else 2
                patch_slice.append(
                    slice(coords[x_idx], coords[x_idx] + patch_size[x_idx])
                )
            else:
                raise ValueError(f"Unrecognised axis '{d}', axes should be in STCZYX.")

        patch_data: NDArray = self._array[tuple(patch_slice)]  # type: ignore
        patch_axes = self._original_axes.replace("S", "").replace("T", "")
        patch_data = reshape_array(patch_data, patch_axes)[0]  # remove first sample dim
        patch = pad_patch(coords, patch_size, self.data_shape, patch_data)

        return patch

    def _get_T_index(self, sample_idx: int) -> int:
        """Get `T` dimension index given `sample_idx`.

        Parameters
        ----------
        sample_idx : int
            Flat sample index (S*T or S).

        Returns
        -------
        int
            Index along the T axis.
        """
        if "T" not in self._original_axes:
            raise ValueError("No 'T' axis specified in original data axes.")
        axis_idx = self._original_axes.index("T")
        dim = self._original_data_shape[axis_idx]

        # new S' = S*T
        # T_idx = S_idx' // T_size
        # S_idx = S_idx' % T_size
        # - floor divide finds the row
        # - modulus finds how far along the row i.e. the column
        return sample_idx % dim

    def _get_S_index(self, sample_idx: int) -> int:
        """Get `S` dimension index given `sample_idx`.

        Parameters
        ----------
        sample_idx : int
            Flat sample index (S*T or S).

        Returns
        -------
        int
            Index along the S axis.
        """
        if "S" not in self._original_axes:
            raise ValueError("No 'S' axis specified in original data axes.")
        if "T" in self._original_axes:
            T_axis_idx = self._original_axes.index("T")
            T_dim = self._original_data_shape[T_axis_idx]

            # new S' = S*T
            # T_idx = S_idx' // T_size
            # S_idx = S_idx' % T_size
            # - floor divide finds the row
            # - modulus finds how far along the row i.e. the column
            return sample_idx // T_dim
        else:
            return sample_idx
