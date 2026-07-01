"""Utilities for reshaping arrays between original and transformed space."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import EllipsisType

import numpy as np
from numpy.typing import NDArray

_REF_ORDER = "STCZYX"
_VALID_AXES = set(_REF_ORDER)


def _validate_axes_and_shape(axes: str, shape: Sequence[int]) -> None:
    """Validate axes and shape.

    Parameters
    ----------
    axes : str
        Axes string of the input data (e.g. "YXC", "STCZYX").
    shape : Sequence[int]
        Shape corresponding to `axes`.

    Raises
    ------
    ValueError
        If axes and shape are not compatible.
    """
    if len(axes) != len(shape):
        raise ValueError(
            f"Axes '{axes}' length ({len(axes)}) does not match shape {shape} length "
            f"({len(shape)})."
        )

    invalid = set(axes) - _VALID_AXES
    if invalid:
        raise ValueError(f"Invalid axis names: {invalid}. Must be from {_VALID_AXES}.")

    if len(set(axes)) != len(axes):
        raise ValueError(f"Duplicate axes in '{axes}'.")

    if "Y" not in axes or "X" not in axes:
        raise ValueError("Axes must contain Y and X.")


@dataclass(frozen=True)
class AxesTransform:
    """Transformation between original and transformed space axes.

    All transformation decisions are derived from the original axes string,
    shape.

    Attributes
    ----------
    original_axes : str
        User defined attribute. Axes string of the input data (e.g. "YXC", "STCZYX").
    original_shape : tuple[int, ...]
        User defined attribute. Shape corresponding to `original_axes`.
    sample_dims : list[str]
        Computed property. Original dimensions merged into S. Spatial (Y, X and Z), as
        well as channels, are never considered sample dimensions.
    original_has_z : bool
        Computed property. Whether original data contains a Z axis.
    original_dim_sizes : dict[str, int]
        Computed property. Map from axis name to original size.
    transformed_axes : str
        Computed property. Transformed axes string: "SC(Z)YX".
    transformed_shape : tuple[int, ...]
        Computed property. Expected shape after forward transformation.
    order_permutation : list[int]
        Computed property. Permutation to reorder original axes to STCZYX reference
        order in SC(Z)YX space.
    """

    original_axes: str
    """Original axes string of the input data (e.g. "YXC", "STCZYX")."""

    original_shape: Sequence[int]
    """Shape corresponding to `original_axes`."""

    def __post_init__(self) -> None:
        """Validate original axes and shape."""
        _validate_axes_and_shape(self.original_axes, self.original_shape)

    @property
    def sample_dims(self) -> list[str]:
        """Original dimensions merged into S.

        Spatial (Y, X and Z), as well as channels, are never considered sample
        dimensions.

        Returns
        -------
        list[str]
            List of original axes that are considered sample dimensions and merged into
            S. Will be in the order they appear in the reference STCZYX.
        """
        excluded = {"C", "Z", "Y", "X"}

        return [a for a in _REF_ORDER if a in self.original_axes and a not in excluded]

    @property
    def c_added_to_original(self) -> bool:
        """C is added as a singleton dimension.

        Returns
        -------
        bool
            True if C is not in original axes, False otherwise.
        """
        return "C" not in self.original_axes

    @property
    def original_has_z(self) -> bool:
        """Original data contains a Z axis.

        Returns
        -------
        bool
            True if Z is in original axes, False otherwise.
        """
        return "Z" in self.original_axes

    @property
    def original_dim_sizes(self) -> dict[str, int]:
        """Map from axis name to original size.

        Returns
        -------
        dict[str, int]
            Dictionary mapping axis name to its size in the original shape.
        """
        return dict(zip(self.original_axes, self.original_shape, strict=True))

    @property
    def transformed_axes(self) -> str:
        """Transformed axes string, `SC(Z)YX` or `SCYX`.

        Returns
        -------
        str
            Transformed axes string. Will be `SCZYX` if original data has Z axis,
            otherwise `SCYX`.
        """
        return "SCZYX" if self.original_has_z else "SCYX"

    @property
    def transformed_shape(self) -> tuple[int, ...]:
        """Expected shape in transformed space.

        Returns
        -------
        tuple[int, ...]
            Expected shape after forward transformation, in the order of
            `transformed_axes`.
        """
        dim_sizes = self.original_dim_sizes

        # resulting sample size is the product of all multiplexed axis sizes
        s = 1
        for dim in self.sample_dims:
            s *= dim_sizes[dim]

        c = dim_sizes.get("C", 1)

        if self.original_has_z:
            return (s, c, dim_sizes["Z"], dim_sizes["Y"], dim_sizes["X"])

        return (s, c, dim_sizes["Y"], dim_sizes["X"])

    @property
    def order_permutation(self) -> list[int]:
        """Permutation to reorder original axes to STCZYX reference order.

        Returns
        -------
        list[int]
            List of indices representing the permutation to reorder original axes to
            STCZYX reference order. Only includes axes present in the original axes.
        """
        return [
            self.original_axes.index(a) for a in _REF_ORDER if a in self.original_axes
        ]

    def calc_original_T_idx(self, sample_idx: int) -> int:
        """Calculate the original index for the `T` dimension given `sample_idx`.

        Parameters
        ----------
        sample_idx : int
            Transformed sample index.

        Returns
        -------
        int
            Index along the T axis.
        """
        if "T" not in self.original_axes:
            raise ValueError("No 'T' axis specified in original data axes.")
        axis_idx = self.original_axes.index("T")
        dim = self.original_shape[axis_idx]

        return sample_idx % dim

    def calc_original_S_idx(self, sample_idx: int) -> int:
        """Calculate the original index for the `S` dimension given `sample_idx`.

        Parameters
        ----------
        sample_idx : int
            Transformed sample index.

        Returns
        -------
        int
            Index along the S axis.
        """
        if "S" not in self.original_axes:
            raise ValueError("No 'S' axis specified in original data axes.")
        if "T" in self.original_axes:
            T_axis_idx = self.original_axes.index("T")
            T_dim = self.original_shape[T_axis_idx]

            return sample_idx // T_dim
        else:
            return sample_idx


@dataclass(frozen=True)
class RestoredAxesTransform:
    """Transformation from transformed space back to original axes order.

    Validation is performed to ensure that the current shape is compatible with the
    original axes. The only exception is the C dimension, which may be absent in the
    original axes but present in the current shape, or inversely, or have a different
    size. In these cases, the resulting C dimension follows the current shape.
    """

    original_axes: str
    """Original axes string of the full data."""

    original_shape: Sequence[int]
    """Original shape of the full data."""

    current_shape: tuple[int, ...]
    """Current transformed shape, either SC(Z)YX or C(Z)YX."""

    current_is_tile: bool = False
    """Whether current_shape is a tile shape (C(Z)YX). This is used to identify the axes
    order in `current_shape`."""

    def __post_init__(self) -> None:
        """Validate current shape and axes."""
        _validate_axes_and_shape(self.original_axes, self.original_shape)

        if self.current_is_tile and len(self.current_shape) not in (3, 4):
            raise ValueError(
                f"Current shape {self.current_shape} is not a valid tile "
                f"shape (C(Z)YX)."
            )
        elif not self.current_is_tile and len(self.current_shape) not in (4, 5):
            raise ValueError(
                f"Current shape {self.current_shape} is not a valid array "
                f"shape (SC(Z)YX)."
            )

        # validate that spatial axes are the same
        if ("Z" in self.original_axes) != ("Z" in self.current_axes):
            raise ValueError(
                f"Original axes {self.original_axes} and current axes "
                f"{self.current_axes} must both contain Z or neither contain Z."
            )

    @property
    def original_dim_sizes(self) -> dict[str, int]:
        """Original dimensions size.

        Returns
        -------
        dict[str, int]
            Dictionary mapping axis name to its size in the original shape.
        """
        return dict(zip(self.original_axes, self.original_shape, strict=True))

    @property
    def sample_dims(self) -> list[str]:
        """Original sample dimensions.

        Returns
        -------
        list[str]
            Original sample dimensions.
        """
        return [a for a in _REF_ORDER if a in self.original_axes and a in "ST"]

    @property
    def current_axes(self) -> str:
        """Current axes in transformed space.

        Returns
        -------
        str
            Axes of the current data in transformed space.

        Raises
        ------
        ValueError
            If the length of the shape is not compatible with the expected length given
            `current_is_tile`.
        """
        match len(self.current_shape):
            case 5:
                return "SCZYX"
            case 4:
                return "CZYX" if self.current_is_tile else "SCYX"
            case 3:
                return "CYX"
            case _:
                raise ValueError(
                    f"Current shape {self.current_shape} is not a valid array or tile "
                    f"shape (SC(Z)YX or C(Z)YX)."
                )

    @property
    def current_c_size(self) -> int:
        """Current number of channels in transformed space.

        Returns
        -------
        int
            Number of channels in the transformed space.
        """
        return self.current_shape[self.current_axes.index("C")]

    @property
    def drop_current_c(self) -> bool:
        """Whether current C should be dropped.

        Returns
        -------
        bool
            True if original data had no C and transformed data has a singleton C
            axis, False otherwise.
        """
        return self.current_c_size == 1

    @property
    def restored_array_axes(self) -> list[str]:
        """Restored axes order for complete arrays with sample dimensions.

        Keeps original axes order, except that if original data had no C and transformed
        data has a non-singleton C axis, then C is inserted just before spatial axes.

        Returns
        -------
        list[str]
            List of axes in the restored data output, following `restored_array_axes`.
        """
        axes = list(self.original_axes)

        if "C" in axes and self.drop_current_c:
            # remove C dimension
            axes.remove("C")
        elif "C" not in axes and self.current_c_size > 1:
            # insert C before the first spatial axis (Z, Y, or X)
            first_spatial_idx = next(i for i, axis in enumerate(axes) if axis in "ZYX")
            axes.insert(first_spatial_idx, "C")

        return axes

    @property
    def restored_axes(self) -> list[str]:
        """Restored axes order for the current output.

        Tiles do not carry S/T dimensions, so S/T axes are removed from the output axes
        order if `current_is_tile` is True.

        Returns
        -------
        list[str]
            List of axes in the restored data output, following `restored_array_axes`.
        """
        axes = self.restored_array_axes
        if self.current_is_tile:
            return [axis for axis in axes if axis not in "ST"]
        return axes

    @property
    def restored_array_shape(self) -> tuple[int, ...]:
        """Shape of the destination array indexed by stitch slices.

        This shape follows `restored_array_axes` and matches original shape except for
        C, which keeps transformed-space channel dimension.

        Returns
        -------
        tuple[int, ...]
            Shape of the destination array indexed by stitch slices, following
            `restored_array_axes`.
        """
        sizes: list[int] = []
        original_sizes = self.original_dim_sizes
        for axis in self.restored_array_axes:
            if axis == "C":
                sizes.append(self.current_c_size)
            else:
                sizes.append(original_sizes[axis])
        return tuple(sizes)

    def _transform_S_and_C(self, data: NDArray) -> tuple[NDArray, list[str]]:
        """Restore transformed axes by unflattening S and dropping singleton new C.

        Parameters
        ----------
        data : numpy.ndarray
            Input data array in transformed space.

        Returns
        -------
        numpy.ndarray
            Data array with S unflattened and singleton new C dropped.
        list[str]
            List of axes in the current transformed array after S unflattening and C
            dropping.
        """
        current_axes = list(self.current_axes)

        if "S" in current_axes:
            sample_dims = self.sample_dims
            if sample_dims:
                sample_sizes = tuple(self.original_dim_sizes[d] for d in sample_dims)
                data = data.reshape(sample_sizes + data.shape[1:])
                current_axes = list(sample_dims) + current_axes[1:]
            else:
                data = data.reshape(data.shape[1:])
                current_axes = current_axes[1:]

        if self.drop_current_c:
            c_idx = current_axes.index("C")
            data = np.squeeze(data, axis=c_idx)
            current_axes.pop(c_idx)

        return data, current_axes

    def _reorder_to_original_axes(
        self,
        data: NDArray,
        current_axes: list[str],
    ) -> NDArray:
        """Reorder data axes to match original order while keeping new axes leading.

        Parameters
        ----------
        data : numpy.ndarray
            Input data array in transformed space.
        current_axes : list[str]
            List of axes in the current transformed array.

        Returns
        -------
        numpy.ndarray
            Data array reordered to match original axes order, with any new axes
            leading.
        """
        target_axes = self.restored_axes
        if current_axes == target_axes:
            return data

        permutation = [current_axes.index(axis) for axis in target_axes]
        return np.transpose(data, permutation)

    def restore(self, data: NDArray) -> NDArray:
        """Restore transformed data to the output layout used by original axes.

        Parameters
        ----------
        data : numpy.ndarray
            Input data array in transformed space.

        Returns
        -------
        numpy.ndarray
            Data array restored to the original axes order.
        """
        data, axes = self._transform_S_and_C(data)
        return self._reorder_to_original_axes(data, axes)

    def stitch_slices(
        self,
        sample_idx: int,
        stitch_coords: Sequence[int],
        crop_size: Sequence[int],
    ) -> tuple[slice | int, ...]:
        """Build slices that index into the restored output dimensions.

        Parameters
        ----------
        sample_idx : int
            Index of the sample in transformed space (S axis) to stitch back.
        stitch_coords : Sequence[int]
            Starting coordinates of the tile in the original spatial axes (Y, X and Z if
            present).
        crop_size : Sequence[int]
            Size of the tile in the original spatial axes (Y, X and Z if present).

        Returns
        -------
        tuple[slice | int, ...]
            Slices that index into the restored output dimensions.
        """
        if len(crop_size) != len(stitch_coords):
            raise ValueError(
                f"Length of `crop_size` ({len(crop_size)}) must match length of "
                f"`stitch_coords` ({len(stitch_coords)})."
            )

        slice_by_axis: dict[str, slice | int] = {}

        # handle sample dimensions (S and/or T)
        sample_dims = self.sample_dims
        if sample_dims:
            sample_dim_sizes = [self.original_dim_sizes[d] for d in sample_dims]
            sample_indices = [
                int(i) for i in np.unravel_index(sample_idx, sample_dim_sizes)
            ]
            for axis, index in zip(sample_dims, sample_indices, strict=True):
                slice_by_axis[axis] = index

        # get full array axes order
        restored_array_axes = self.restored_array_axes

        # add C slice if C is present in the final array
        if "C" in restored_array_axes:
            slice_by_axis["C"] = slice(0, self.current_c_size)

        # coordinates are provided in transformed-space spatial order
        transformed_spatial_axes = [axis for axis in self.current_axes if axis in "ZYX"]
        if len(transformed_spatial_axes) != len(crop_size):
            raise ValueError(
                "Length of `crop_size` must match the spatial dimensions of the "
                f"current transformed data ({len(transformed_spatial_axes)})."
            )

        transformed_spatial_slices: dict[str, slice] = {}
        for axis, start, length in zip(
            transformed_spatial_axes, stitch_coords, crop_size, strict=True
        ):
            transformed_spatial_slices[axis] = slice(start, start + length)

        for axis in restored_array_axes:
            if axis in transformed_spatial_slices:
                slice_by_axis[axis] = transformed_spatial_slices[axis]

        # return slices ordered by restored_axes
        return tuple(
            slice_by_axis[axis] for axis in restored_array_axes if axis in slice_by_axis
        )

    def adjust_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
        """Adjust shape to match the restored array shape.

        This method adjusts the input shape to match the restored array shape, taking
        into account any differences in the C dimension. If the original data had no C
        and the transformed data has a non-singleton C axis, then the C dimension is
        inserted just before spatial axes. If the original data had a C axis and the
        transformed data has a singleton C axis, then the C dimension is removed.

        Parameters
        ----------
        shape : Sequence[int]
            Input shape to adjust.

        Returns
        -------
        tuple[int, ...]
            Adjusted shape that matches the restored array shape.
        """
        if len(shape) != len(self.original_shape):
            raise ValueError(
                f"Input shape {shape} does not match original shape "
                f"{self.original_shape} length."
            )

        axes = self.original_axes
        adjusted_shape = list(shape)

        if "C" in axes and self.drop_current_c:
            # remove C dimension
            adjusted_shape.pop(axes.index("C"))
        elif "C" not in axes and self.current_c_size > 1:
            # insert C before the first spatial axis (Z, Y, or X)
            first_spatial_idx = next(i for i, axis in enumerate(axes) if axis in "ZYX")
            adjusted_shape.insert(first_spatial_idx, 1)

        return tuple(adjusted_shape)


def reshape_array(
    array: NDArray,
    original_axes: str,
) -> NDArray:
    """Reshape array from arbitrary axes order to `SC(Z)YX`.

    Parameters
    ----------
    array : numpy.ndarray
        Input array.
    original_axes : str
        Original axes string describing current dimension order (e.g. `YXC`).

    Returns
    -------
    numpy.ndarray
        Array reshaped to `SC(Z)YX`.
    """
    transform = AxesTransform(original_axes, array.shape)

    # reorder axes to reference STCZYX
    permutation = transform.order_permutation
    array = np.moveaxis(array, permutation, list(range(len(permutation))))

    # merge sample dims
    n_sample = len(transform.sample_dims)
    if n_sample > 1:
        array = array.reshape((-1,) + array.shape[n_sample:])
    elif n_sample == 0:
        array = array[np.newaxis]

    # add singleton C, if necessary
    if transform.c_added_to_original:
        array = np.expand_dims(array, 1)

    return array


def reshape_patch(
    patch: NDArray,
    original_axes: str,
) -> NDArray:
    """Reshape patch from arbitrary axes order to `C(Z)YX`.

    Parameters
    ----------
    patch : numpy.ndarray
        Input patch, patches do not include the `"S"` or `"T"` dimension.
    original_axes : str
        Axes string that describes the original dimensions of the data the patch was
        sampled from, (e.g. SYXC).

    Returns
    -------
    numpy.ndarray
        Patch reshaped to `C(Z)YX`.
    """
    # remove S and T from axes to get patch axes
    patch_axes = original_axes.replace("S", "").replace("T", "")
    patch_data = reshape_array(patch, patch_axes)[0]  # remove first sample dim
    return patch_data


def restore_array(
    array: NDArray,
    original_axes: str,
    original_shape: Sequence[int],
) -> NDArray:
    """Restore array from `SC(Z)YX` space back to original axes and shape.

    If `array` has different spatial dimensions or number of channels than the original
    array, then the restored array will have the same shape as `array` in those
    dimensions, but will still be reordered to match the original axes order.

    Parameters
    ----------
    array : numpy.ndarray
        Array in `SC(Z)YX` format.
    original_axes : str
        Original axes string (e.g. `YXC`).
    original_shape : Sequence[int]
        Original shape of the data.

    Returns
    -------
    numpy.ndarray
        Array with original axes order and shape restored.

    Raises
    ------
    ValueError
        If input array is not 4D (SCYX) or 5D (SCZYX), or if restoring shape is not
        supported for the given original axes (e.g. T as Z with CZI format).
    """
    if len(array.shape) not in (4, 5):
        raise ValueError(f"Expected 4D (SCYX) or 5D (SCZYX), got {len(array.shape)}D.")

    if len(array.shape) == 5 and original_axes == "SCTYX":
        raise ValueError(
            "Restoring shape is currently not supported for CZI format (T used as "
            "depth axis)."
        )

    transform = RestoredAxesTransform(
        original_axes=original_axes,
        original_shape=original_shape,
        current_shape=array.shape,
        current_is_tile=False,
    )
    return transform.restore(array)


def restore_tile(
    tile: NDArray,
    original_axes: str,
    original_shape: Sequence[int],
) -> NDArray:
    """Restore single tile from `C(Z)YX` space back to original axes and shape.

    Parameters
    ----------
    tile : numpy.ndarray
        Tile in `C(Z)YX` format.
    original_axes : str
        Original axes string of the full data.
    original_shape : Sequence[int]
        Original shape of the full data.

    Returns
    -------
    numpy.ndarray
        Tile with original spatial axes order restored.
    """
    if len(tile.shape) not in (3, 4):
        raise ValueError(f"Expected 3D (CYX) or 4D (CZYX), got {len(tile.shape)}D.")

    transform = RestoredAxesTransform(
        original_axes=original_axes,
        original_shape=original_shape,
        current_shape=tile.shape,
        current_is_tile=True,
    )
    return transform.restore(tile)


# TODO delete
def get_stitch_slices(
    original_axes: str,
    original_shape: Sequence[int],
    tile_shape: Sequence[int],
    sample_idx: int,
    stitch_coords: Sequence[int],
    crop_size: Sequence[int],
) -> tuple[slice | int, ...]:
    """Get slices to stitch tile back into original array.

    `sample_idx and `stitch_coords` are expressed with respect to the transformed space
    (SCZYX or SCYX). The returned slices will index into the array in original space to
    stitch the tile back in place.

    Note that the array in which the tile will be indexed may have different C and
    spatial dimensions and dimension sizes.

    Parameters
    ----------
    original_axes : str
        Original axes string of the full data.
    original_shape : Sequence[int]
        Original shape of the full data.
    tile_shape : Sequence[int]
        Shape of the tile in transformed space (C(Z)YX or CYX).
    sample_idx : int
        Index of the sample in transformed space (S axis) to stitch back.
    stitch_coords : Sequence[int]
        Starting coordinates of the tile in the original spatial axes (Y, X and Z if
        present).
    crop_size : Sequence[int]
        Size of the tile in the original spatial axes (Y, X and Z if present).

    Returns
    -------
    tuple[slice | int, ...]
        Slices to index into the original array for stitching the tile back in place.
    """
    if len(crop_size) != len(stitch_coords):
        raise ValueError(
            f"Length of `crop_size` ({len(crop_size)}) must match length of "
            f"`stitch_coords` ({len(stitch_coords)})."
        )
    if len(crop_size) != len(tile_shape) - 1:
        raise ValueError(
            f"Length of `crop_size` ({len(crop_size)}) must match spatial dimensions "
            f"of `tile_shape` ({len(tile_shape) - 1})."
        )

    transform = RestoredAxesTransform(
        original_axes=original_axes,
        original_shape=original_shape,
        current_shape=tuple(tile_shape),
        current_is_tile=True,
    )
    return transform.stitch_slices(sample_idx, stitch_coords, crop_size)


# TODO delete
def get_restored_array_shape(
    original_axes: str,
    original_shape: Sequence[int],
    current_shape: Sequence[int],
) -> tuple[int, ...]:
    """Get the shape of the restored array in original axes order.

    Parameters
    ----------
    original_axes : str
        Original axes string of the full data.
    original_shape : Sequence[int]
        Original shape of the full data.
    current_shape : Sequence[int]
        Current shape of the array.

    Returns
    -------
    tuple[int, ...]
        Shape of the restored array in original axes order.
    """
    transform = RestoredAxesTransform(
        original_axes=original_axes,
        original_shape=original_shape,
        current_shape=tuple(current_shape),
        current_is_tile=True,
    )
    return transform.restored_array_shape


def get_patch_slices(
    original_axes: str,
    original_shape: Sequence[int],
    sample_idx: int,
    channels: Sequence[int] | None,
    coords: Sequence[int],
    patch_size: Sequence[int],
) -> tuple[slice | int | Sequence[int] | EllipsisType, ...]:
    """Get slices to extract patch from an array.

    The argument `original_axes` describes the dimension order of the array.

    `sample_idx` is expressed with respect to the transformed space where the `"S"` and
    `"T"` dimensions are flattened together, if both or either are present.

    Parameters
    ----------
    original_axes : str
        Original axes string of the full data.
    original_shape : Sequence[int]
        Original shape of the full data.
    sample_idx : int
        Index of the sample in transformed space (S axis) to stitch back.
    channels : sequence of int or None
        Channel indices to extract. If `None`, all channels will be extracted.
    coords : Sequence[int]
        Starting coordinates of the patch in the original spatial axes (Y, X and Z if
        present).
    patch_size : Sequence[int]
        Size of the patch in the spatial axes (Y, X and Z if present).

    Returns
    -------
    tuple[slice | int | Sequence[int] | EllipsisType, ...]
        Slices to index into the original array to extract the patch.
    """
    transform = AxesTransform(original_axes, tuple(original_shape))
    transformed_shape = transform.transformed_shape
    # original axes assumed to be any subset of STCZYX (containing YX), in any order
    # arguments must be transformed to index data in original axes order
    # to do this: loop through original axes and append correct index/slice
    #   for each case: STCZYX
    #   Note: if any axis is not present in original_axes it is skipped.

    # guard for no S and T in original axes
    if ("S" not in original_axes) and ("T" not in original_axes):
        if sample_idx not in [0, -1]:
            raise IndexError(
                f"Sample index {sample_idx} out of bounds for S axes with size 1."
            )

    # check that channels are within bounds
    if channels is not None:
        max_channel = transformed_shape[1] - 1  # channel is second dimension
        for ch in channels:
            if ch > max_channel:
                raise ValueError(
                    f"Channel index {ch} is out of bounds for data with "
                    f"{transformed_shape[1]} channels. Check the provided `channels` "
                    f"parameter in the configuration for erroneous channel "
                    f"indices."
                )

    patch_slice: list[slice | int | Sequence[int] | EllipsisType] = []
    for d in original_axes:
        if d == "S":
            patch_slice.append(transform.calc_original_S_idx(sample_idx))
        elif d == "T":
            patch_slice.append(transform.calc_original_T_idx(sample_idx))
        elif d == "C":
            patch_slice.append(channel_slice(channels))
        elif d == "Z":
            patch_slice.append(slice(coords[0], coords[0] + patch_size[0]))
        elif d == "Y":
            y_idx = 0 if "Z" not in original_axes else 1
            patch_slice.append(slice(coords[y_idx], coords[y_idx] + patch_size[y_idx]))
        elif d == "X":
            x_idx = 1 if "Z" not in original_axes else 2
            patch_slice.append(slice(coords[x_idx], coords[x_idx] + patch_size[x_idx]))
        else:
            raise ValueError(f"Unrecognized axis '{d}', axes should be in STCZYX.")

    return tuple(patch_slice)


def channel_slice(
    channels: Sequence[int] | None,
) -> EllipsisType | Sequence[int]:
    """Create a slice or sequence for indexing channels while preserving dimensions.

    Parameters
    ----------
    channels : Sequence[int] | None
        The channel indices to select, or None to select all channels.

    Returns
    -------
    EllipsisType | Sequence[int]
        An indexing object that can be used to index the channel dimension while
        preserving it.
    """
    if channels is None:
        return ...

    if len(channels) == 0:
        raise ValueError("Channel index sequence cannot be empty.")

    return channels
