from collections.abc import Sequence
from typing import Union

import zarr
from numpy.typing import NDArray

from careamics.dataset.dataset_utils import reshape_array

from .utils import pad_patch, reshaped_array_shape


class ZarrImageStack:
    """
    A class for extracting patches from an image stack that is stored as a zarr array.
    """

    def __init__(self, group: zarr.Group, data_path: str, axes: str):
        if not isinstance(group, zarr.Group):
            raise TypeError(f"group must be a zarr.Group instance, got {type(group)}.")

        self._group = group
        self._array = group[data_path]
        self._store = str(group.store_path)
        self._source = self._array.store_path

        # TODO: validate axes
        #   - must contain XY
        #   - must be subset of STCZYX
        self._original_axes = axes
        self._original_data_shape: tuple[int, ...] = self._array.shape
        self.data_shape = reshaped_array_shape(axes, self._original_data_shape)
        self.data_dtype = self._array.dtype
        self._chunk_size = self._array.chunks

    # Used to identify the source of the data and write to similar path during pred
    @property
    def source(self) -> str:
        # e.g. file://data/bsd68.zarr/train/
        return self._source

    @property
    def chunk_size(self) -> Sequence[int]:
        return self._chunk_size

    def extract_patch(
        self, sample_idx: int, coords: Sequence[int], patch_size: Sequence[int]
    ) -> NDArray:
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

        patch_slice: list[Union[int, slice]] = []
        for d in self._original_axes:
            if d == "S":
                patch_slice.append(self._get_S_index(sample_idx))
            elif d == "T":
                patch_slice.append(self._get_T_index(sample_idx))
            elif d == "C":
                patch_slice.append(slice(None, None))
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

        patch_data = self._array[tuple(patch_slice)]
        patch_axes = self._original_axes.replace("S", "").replace("T", "")
        patch_data = reshape_array(patch_data, patch_axes)[0]  # remove first sample dim
        patch = pad_patch(coords, patch_size, self.data_shape, patch_data)

        return patch

    def _get_T_index(self, sample_idx: int) -> int:
        """Get T index given `sample_idx`."""
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
        """Get S index given `sample_idx`."""
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
