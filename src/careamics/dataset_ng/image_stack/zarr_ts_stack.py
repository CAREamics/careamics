"""TensorStore-based image stack for zarr arrays."""

from collections.abc import Sequence
from pathlib import Path

import tensorstore as ts
from numpy.typing import DTypeLike, NDArray

from careamics.dataset.dataset_utils import reshape_array

from .image_utils.image_stack_utils import channel_slice, pad_patch, reshape_array_shape


class ZarrTSImageStack:
    """
    Faster image stack implementation for zarr arrays.

    This implementation uses the TensorStore library for reading data.

    Parameters
    ----------
    store_path : str | Path
        Path to the zarr store (the .zarr directory).
    data_path : str
        Path to the array within the zarr store (e.g., "data").
    axes : str
        Axes string describing the data layout (subset of STCZYX).

    Attributes
    ----------
    source : str
        Source path/URI of the zarr array.
    chunks : Sequence[int]
        Chunk sizes in the order of data_shape (SC(Z)YX).
    shards : Sequence[int] | None
        Shard sizes in the order of data_shape (SC(Z)YX), or None if
        the array is not sharded.
    data_dtype : DTypeLike
        Data type of the array.
    """

    def __init__(self, store_path: str | Path, data_path: str, axes: str):
        """
        Initialize a TensorStore-based zarr image stack.

        Parameters
        ----------
        store_path : str | Path
            Path to the zarr store (the .zarr directory).
        data_path : str
            Path to the array within the zarr store (e.g., "data").
        axes : str
            Axes string describing the data layout (subset of STCZYX).

        Raises
        ------
        ValueError
            If the array cannot be opened with TensorStore.
        """
        self._store = str(store_path)
        self._data_path = data_path

        try:
            self._array = ts.open(
                {
                    # specs: https://google.github.io/tensorstore/driver/zarr3/
                    "driver": "zarr3",
                    "kvstore": {
                        "driver": "file",
                        "path": self._store,  # needs to be zarr store root
                    },
                    "path": data_path,  # path to array within the store
                    "recheck_cached_data": False,  # assume data does not change
                }
            ).result()
        except Exception as e:
            raise ValueError(
                f"Failed to open zarr array at '{self._store}/{data_path}' with "
                f"TensorStore: {e}"
            ) from e

        self._source = f"{self._store}/{data_path}"

        # TODO: validate axes
        #   - must contain XY
        #   - must be subset of STCZYX
        self._original_axes = axes
        self._original_data_shape: tuple[int, ...] = tuple(self._array.shape)
        self.data_shape = reshape_array_shape(axes, self._original_data_shape)
        self._data_dtype = self._array.dtype.numpy_dtype

        # extract chunk and shard sizes
        chunk_layout = self._array.chunk_layout
        if chunk_layout.read_chunk.shape:
            self._chunk_size = reshape_array_shape(
                axes, tuple(chunk_layout.read_chunk.shape), add_singleton=False
            )
        else:
            # no chunk available
            # delegate to writer the responsibility to set chunking
            # (e.g. writes_tile_zarr with auto chunking)
            self._chunk_size = None

        self._shard_size = None

    @property
    def source(self) -> str:
        """Source path to the zarr array."""
        return str(self._source)

    @property
    def chunks(self) -> Sequence[int]:
        """Chunks size in the order of data_shape (SC(Z)YX)."""
        return self._chunk_size

    @property
    def shards(self) -> Sequence[int] | None:
        """Shard size in the order of data_shape (SC(Z)YX)."""
        return self._shard_size

    @property
    def data_dtype(self) -> DTypeLike:
        """Data type of the array."""
        return self._data_dtype

    def extract_patch(
        self, sample_idx: int, coords: Sequence[int], patch_size: Sequence[int]
    ) -> NDArray:
        """Extract a patch from the array without channel selection.

        Parameters
        ----------
        sample_idx : int
            Sample index.
        coords : Sequence[int]
            Starting coordinates in (Z)YX order.
        patch_size : Sequence[int]
            Size of the patch in (Z)YX order.

        Returns
        -------
        NDArray
            Extracted patch in C(Z)YX format.
        """
        return self.extract_channel_patch(sample_idx, None, coords, patch_size)

    def extract_channel_patch(
        self,
        sample_idx: int,
        channels: Sequence[int] | None,
        coords: Sequence[int],
        patch_size: Sequence[int],
    ) -> NDArray:
        """Extract a patch with optional channel selection.

        Parameters
        ----------
        sample_idx : int
            Sample index.
        channels : Sequence[int] | None
            Channels to extract. If None, all channels are extracted.
        coords : Sequence[int]
            Starting coordinates in (Z)YX order.
        patch_size : Sequence[int]
            Size of the patch in (Z)YX order.

        Returns
        -------
        NDArray
            Extracted patch in C(Z)YX format.
        """
        # Guard for no S and T in original axes
        if ("S" not in self._original_axes) and ("T" not in self._original_axes):
            if sample_idx not in [0, -1]:
                raise IndexError(
                    f"Sample index {sample_idx} out of bounds for S axes with size "
                    f"{self.data_shape[0]}"
                )

        # Check that channels are within bounds
        if channels is not None:
            max_channel = self.data_shape[1] - 1
            for ch in channels:
                if ch > max_channel:
                    raise ValueError(
                        f"Channel index {ch} is out of bounds for data with "
                        f"{self.data_shape[1]} channels. Check the provided `channels` "
                        f"parameter in the configuration for erroneous channel "
                        f"indices."
                    )

        # Build indexing expression for TensorStore
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

        # Read data using TensorStore (synchronous read)
        patch_data: NDArray = self._array[tuple(patch_slice)].read().result()

        # Reshape to standard format
        patch_axes = self._original_axes.replace("S", "").replace("T", "")
        patch_data = reshape_array(patch_data, patch_axes)[0]  # remove first sample dim

        # Pad if needed (e.g., at image boundaries)
        patch = pad_patch(coords, patch_size, self.data_shape, patch_data)

        return patch

    def _get_T_index(self, sample_idx: int) -> int:
        """Get T index given `sample_idx`."""
        if "T" not in self._original_axes:
            raise ValueError("No 'T' axis specified in original data axes.")
        axis_idx = self._original_axes.index("T")
        dim = self._original_data_shape[axis_idx]

        # new S' = S*T
        # T_idx = S_idx' % T_size
        return sample_idx % dim

    def _get_S_index(self, sample_idx: int) -> int:
        """Get S index given `sample_idx`."""
        if "S" not in self._original_axes:
            raise ValueError("No 'S' axis specified in original data axes.")
        if "T" in self._original_axes:
            T_axis_idx = self._original_axes.index("T")
            T_dim = self._original_data_shape[T_axis_idx]

            # new S' = S*T
            # S_idx = S_idx' // T_size
            return sample_idx // T_dim
        else:
            return sample_idx
