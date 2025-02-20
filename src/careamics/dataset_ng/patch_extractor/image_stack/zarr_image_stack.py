from collections.abc import Sequence
from pathlib import Path
from typing import Union

import zarr
import zarr.storage
from numpy.typing import NDArray
from typing_extensions import Self

from careamics.dataset.dataset_utils import reshape_array


class ZarrImageStack:
    """
    A class for extracting patches from an image stack that is stored as a zarr array.
    """

    # TODO: keeping store type narrow so that it has the path attribute
    #   base zarr store is zarr.storage.Store, includes MemoryStore
    def __init__(self, store: zarr.storage.FSStore, data_path: str, axes: str):
        self._store = store
        self._array = zarr.open_array(store=self._store, path=data_path, mode="r")
        self._original_axes = axes  # TODO: validate axes
        self._original_data_shape: tuple[int, ...] = self._array.shape
        self.data_shape = _reshaped_array_shape(axes, self._original_data_shape)

    # TODO: not sure if this is useful
    @property
    def source(self) -> Path:
        return Path(self._store.path) / self._array.path

    # automatically finds axes from metadata
    @classmethod
    def from_ome_zarr(cls, path: Union[Path, str]) -> Self:
        """
        Will only use the first resolution in the hierarchy.

        Assumes the path only contains 1 image.

        Path can be to a local file, or it can be a URL to a zarr stored in the cloud.
        """
        store = zarr.storage.FSStore(url=path)
        group = zarr.open_group(store=store, mode="r")
        if "multiscales" not in group.attrs:
            raise ValueError(
                f"Zarr at path '{path}' cannot be loaded as an OME-Zarr because it "
                "does not contain the attribute 'multiscales'."
            )
        # TODO: why is this a list of length 1, 0 index also in ome-zarr-python
        multiscales_metadata = group.attrs["multiscales"][0]

        # get axes
        axes_list = [axes_data["name"] for axes_data in multiscales_metadata["axes"]]
        axes = "".join(axes_list).upper()

        first_multiscale_path = multiscales_metadata["datasets"][0]["path"]

        return cls(store=store, data_path=first_multiscale_path, axes=axes)

    def extract_patch(
        self, sample_idx: int, coords: Sequence[int], patch_size: Sequence[int]
    ) -> NDArray:
        # reorder slice to index original data shape
        patch_slice: list[Union[int, slice]] = []
        for d in self._original_axes:
            if d == "S":
                patch_slice.append(self._get_S_index(sample_idx))
            elif d == "T":
                patch_slice.append(self._get_T_index(sample_idx))
            elif d == "C":
                patch_slice.append(slice(None, None))
            elif d == "Z":
                patch_slice.append(self._get_Z_slice(coords, patch_size))
            elif d == "Y":
                patch_slice.append(self._get_Y_slice(coords, patch_size))
            elif d == "X":
                patch_slice.append(self._get_X_slice(coords, patch_size))
            else:
                raise ValueError(f"Unrecognised axis '{d}', axes should be in STCZYX.")

        patch = self._array[tuple(patch_slice)]
        patch_axes = self._original_axes.replace("S", "").replace("T", "")
        return reshape_array(patch, patch_axes)[0]  # remove first sample dim

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
        return sample_idx // dim

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
            return sample_idx % T_dim
        else:
            return sample_idx

    def _get_Z_slice(self, coords: Sequence[int], patch_size: Sequence[int]) -> slice:
        """Get z slice given `coords` and `patch_size`"""
        if "Z" not in self._original_axes:
            raise ValueError("No 'Z' axis specified in original data axes.")
        idx = 0
        return slice(coords[idx], coords[idx] + patch_size[idx])

    def _get_Y_slice(self, coords: Sequence[int], patch_size: Sequence[int]) -> slice:
        """Get y slice given `coords` and `patch_size`"""
        idx = 0 if "Z" not in self._original_axes else 1
        return slice(coords[idx], coords[idx] + patch_size[idx])

    def _get_X_slice(self, coords: Sequence[int], patch_size: Sequence[int]) -> slice:
        """Get x slice given `coords` and `patch_size`"""
        idx = 1 if "Z" not in self._original_axes else 2
        return slice(coords[idx], coords[idx] + patch_size[idx])


# TODO: move to dataset_utils, better name?
def _reshaped_array_shape(axes: str, shape: Sequence[int]) -> tuple[int, ...]:
    """Find resulting shape if reshaping array with given `axes` and `shape`."""
    target_axes = "SCZYX"
    target_shape = []
    for d in target_axes:
        if d in axes:
            idx = axes.index(d)
            target_shape.append(shape[idx])
        elif (d != axes) and (d != "Z"):
            target_shape.append(1)
        else:
            pass

    if "T" in axes:
        idx = axes.index("T")
        target_shape[0] = target_shape[0] * shape[idx]

    return tuple(target_shape)
