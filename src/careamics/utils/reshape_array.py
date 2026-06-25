"""Utilities for reshaping arrays between original and transformed space."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import EllipsisType

import numpy as np
from numpy.typing import NDArray

_REF_ORDER = "STCZYX"
_VALID_AXES = set(_REF_ORDER)


# TODO
# Actually, the shape may change between input and ouputs:
# - number of channels may be different
# - new channels may be added
# - number of spatial dimensions may be different (e.g. 2D vs 3D)


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
    c_added_to_original : bool
        Computed property. Whether C is added as a singleton.
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

    current_shape: tuple[int, ...] | None = None
    """Current shape of the data. Must be in transformed space, either as a full array
    of dims SC(Z)YX or a tile of shape C(Z)YX. If None, will be set to
    `original_shape`."""

    current_is_tile: bool = False
    """Whether the current shape is a tile (C(Z)YX) or a full array (SC(Z)YX). If True,
    `current_shape` must be in tile format."""

    def __post_init__(self) -> None:
        """Validate original axes and shape."""
        if self.current_shape is None:
            self.current_shape = tuple(self.original_shape)
        else:
            if self.current_is_tile:
                if len(self.current_shape) not in (3, 4):
                    raise ValueError(
                        f"Current shape {self.current_shape} is not a valid tile "
                        f"shape (C(Z)YX)."
                    )
            else:
                if len(self.current_shape) not in (4, 5):
                    raise ValueError(
                        f"Current shape {self.current_shape} is not a valid array "
                        f"shape (SC(Z)YX)."
                    )

        if len(self.original_axes) != len(self.original_shape):
            raise ValueError(
                f"Axes '{self.original_axes}' length ({len(self.original_axes)}) "
                f"does not match shape {self.original_shape} length "
                f"({len(self.original_shape)})."
            )

        invalid = set(self.original_axes) - _VALID_AXES
        if invalid:
            raise ValueError(
                f"Invalid axis names: {invalid}. Must be from {_VALID_AXES}."
            )

        if len(set(self.original_axes)) != len(self.original_axes):
            raise ValueError(f"Duplicate axes in '{self.original_axes}'.")

        if "Y" not in self.original_axes or "X" not in self.original_axes:
            raise ValueError("Axes must contain Y and X.")

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
    def current_axes(self) -> str:
        """Current axes.

        Returns
        -------
        str
            Current axes string, based on current shape. Will be a subset of
            `transformed_axes`.
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
    def current_added_axes(self) -> list[str]:
        """Added axes in the current.

        Returns
        -------
        list[str]
            List of axes that are present in the current shape but not in the original
            axes.
        """
        return [
            a
            for a in self.current_axes
            if (
                a not in self.original_axes
                and self.current_shape[self.current_axes.index(a)] > 1
                and a not in "ST"  # exclude sample dims
            )
        ]

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

        # new S' = S*T
        # T_idx = S_idx' // T_size
        # S_idx = S_idx' % T_size
        # - floor divide finds the row
        # - modulus finds how far along the row i.e. the column
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

            # new S' = S*T
            # T_idx = S_idx' // T_size
            # S_idx = S_idx' % T_size
            # - floor divide finds the row
            # - modulus finds how far along the row i.e. the column
            return sample_idx // T_dim
        else:
            return sample_idx


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


def _count_new_axes(
    current_axes: list[str],
    current_shape: Sequence[int],
    original_axes: str,
) -> int:
    """Count non-sample axes that were added in transformed output."""
    return sum(
        1
        for idx, axis in enumerate(current_axes)
        if axis not in original_axes and axis not in "ST" and current_shape[idx] > 1
    )


def _reorder_to_original_axes(
    data: NDArray,
    current_axes: list[str],
    original_axes: str,
) -> NDArray:
    """Reorder data axes to match original order while keeping new axes leading."""
    current_str = "".join(current_axes)
    if current_str == original_axes:
        return data

    n_new_axes = _count_new_axes(current_axes, data.shape, original_axes)
    source = [current_str.index(a) for a in original_axes if a in current_axes]
    destination = list(range(n_new_axes, n_new_axes + len(source)))
    return np.moveaxis(data, source, destination)


def _restore_from_transformed(
    data: NDArray,
    original_axes: str,
    original_shape: Sequence[int],
    *,
    is_tile: bool,
) -> NDArray:
    """Shared restore implementation for full arrays and tiles."""
    transform = AxesTransform(
        original_axes=original_axes,
        original_shape=tuple(original_shape),
        current_shape=data.shape,
        current_is_tile=is_tile,
    )
    current_axes = list(transform.current_axes)

    if not is_tile:
        # Restore sample dimensions from flattened S.
        sample_dims = transform.sample_dims
        sample_sizes = tuple(transform.original_dim_sizes[d] for d in sample_dims)
        data = data.reshape(sample_sizes + data.shape[1:])
        current_axes = list(sample_dims) + current_axes[1:]

    # Remove singleton C only if C did not exist in original axes.
    if transform.c_added_to_original:
        c_idx = 0 if is_tile else 1
        if data.shape[c_idx] == 1:
            data = np.squeeze(data, axis=c_idx)
            current_axes.pop(c_idx)

    return _reorder_to_original_axes(data, current_axes, original_axes)


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

    return _restore_from_transformed(
        data=array,
        original_axes=original_axes,
        original_shape=original_shape,
        is_tile=False,
    )


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

    return _restore_from_transformed(
        data=tile,
        original_axes=original_axes,
        original_shape=original_shape,
        is_tile=True,
    )


def get_original_stitch_slices(
    original_axes: str,
    original_shape: Sequence[int],
    sample_idx: int,
    stitch_coords: Sequence[int],
    crop_size: Sequence[int],
) -> tuple[slice | int, ...]:
    """Get slices to stitch tile back into original array.

    `sample_idx and `stitch_coords` are expressed with respect to the transformed space
    (SCZYX or SCYX). The returned slices will index into the original array for
    stitching the tile back in place.

    Parameters
    ----------
    original_axes : str
        Original axes string of the full data.
    original_shape : Sequence[int]
        Original shape of the full data.
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
    transform = AxesTransform(original_axes, tuple(original_shape))

    stitch_slices: list[slice | int] = []
    which_axes = []

    # unravel sample indices
    if len(transform.sample_dims) >= 1:
        sample_dims = transform.sample_dims
        sample_dim_sizes = [transform.original_dim_sizes[d] for d in sample_dims]
        sample_indices = [
            int(i) for i in np.unravel_index(sample_idx, sample_dim_sizes)
        ]
        stitch_slices.extend(sample_indices)
        which_axes.extend(sample_dims)

    if not transform.c_added_to_original:
        stitch_slices.append(slice(0, transform.original_dim_sizes["C"]))
        which_axes.append("C")

    # add spatial slices
    stitch_slices.extend(
        [
            slice(start, start + length)
            for start, length in zip(stitch_coords, crop_size, strict=True)
        ]
    )
    which_axes.extend([a for a in transform.transformed_axes if a in "ZYX"])
    assert len(stitch_slices) == len(transform.original_axes)

    # reorder slices
    stitch_slices = [
        stitch_slices[which_axes.index(a)] for a in transform.original_axes
    ]

    return tuple(stitch_slices)


# TODO: unify with get_original_stitch_slices
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
