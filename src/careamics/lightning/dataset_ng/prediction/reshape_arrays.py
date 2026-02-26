"""Axes transformation utilities for reshaping arrays between original and DL space."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_REF_ORDER = "STCZYX"
_VALID_AXES = set(_REF_ORDER)


@dataclass(frozen=True)
class AxesTransform:
    """Transformation between original and DL-space axes.

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
    s_added : bool
        Computed property. Whether S is added as a singleton.
    c_added : bool
        Computed property. Whether C is added as a singleton.
    has_z : bool
        Computed property. Whether original data contains a Z axis.
    t_becomes_s : bool
        Computed property. Whether T is the sole sample dim, relabeled as S.
    st_merged : bool
        Computed property. Whether S and T are both sample dims and merged together.
    dim_sizes : dict[str, int]
        Computed property. Map from axis name to original size.
    dl_axes : str
        Computed property. DL axes string: "SC(Z)YX".
    dl_shape : tuple[int, ...]
        Computed property. Expected shape after forward transformation.
    order_permutation : list[int]
        Computed property. Permutation to reorder original axes to STCZYX reference
        order in DL space.
    """

    original_axes: str
    original_shape: Sequence[int]

    def __post_init__(self) -> None:
        """Validate original axes and shape."""
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
    def s_added(self) -> bool:
        """S is added as a singleton dimension.

        Returns
        -------
        bool
            True if there are no sample dimensions and S is added as a singleton, False
            otherwise.
        """
        return len(self.sample_dims) == 0

    @property
    def c_added(self) -> bool:
        """C is added as a singleton dimension.

        Returns
        -------
        bool
            True if C is not in original axes, False otherwise.
        """
        return "C" not in self.original_axes

    @property
    def has_z(self) -> bool:
        """Original data contains a Z axis.

        Returns
        -------
        bool
            True if Z is in original axes, False otherwise.
        """
        return "Z" in self.original_axes

    @property
    def t_becomes_s(self) -> bool:
        """T is the sole sample dim, relabeled as S.

        Returns
        -------
        bool
            True if T is the only sample dimension, False otherwise.
        """
        return self.sample_dims == ["T"]

    @property
    def st_merged(self) -> bool:
        """S and T are both sample dimensions and are merged together.

        Returns
        -------
        bool
            True if both S and T are in sample_dims, False otherwise.
        """
        return "S" in self.sample_dims and "T" in self.sample_dims

    @property
    def dim_sizes(self) -> dict[str, int]:
        """Map from axis name to original size.

        Returns
        -------
        dict[str, int]
            Dictionary mapping axis name to its size in the original shape.
        """
        return dict(zip(self.original_axes, self.original_shape, strict=True))

    @property
    def dl_axes(self) -> str:
        """DL axes string, `SC(Z)YX` or `SCYX`.

        Returns
        -------
        str
            DL axes string. Will be `SCZYX` if original data has Z axis, otherwise
            `SCYX`.
        """
        return "SCZYX" if self.has_z else "SCYX"

    @property
    def dl_shape(self) -> tuple[int, ...]:
        """Expected shape in DL space.

        Returns
        -------
        tuple[int, ...]
            Expected shape after forward transformation, in the order of `dl_axes`.
        """
        dim_sizes = self.dim_sizes

        # resulting sample size is the product of all multiplexed axis sizes
        s = 1
        for dim in self.sample_dims:
            s *= dim_sizes[dim]

        c = dim_sizes.get("C", 1)

        if self.has_z:
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


def reshape_array(
    array: NDArray,
    original_axes: str,
    original_shape: Sequence[int],
) -> NDArray:
    """Reshape array from arbitrary axes order to `SC(Z)YX`.

    Parameters
    ----------
    array : numpy.ndarray
        Input array.
    original_axes : str
        Original axes string describing current dimension order (e.g. `YXC`).
    original_shape : Sequence[int]
        Original shape of the array.

    Returns
    -------
    numpy.ndarray
        Array reshaped to `SC(Z)YX`.
    """
    transform = AxesTransform(original_axes, original_shape)

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
    if transform.c_added:
        array = np.expand_dims(array, 1)

    return array


def restore_array(
    array: NDArray,
    original_axes: str,
    original_shape: Sequence[int],
) -> NDArray:
    """Restore array from `SC(Z)YX` space back to original axes and shape.

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

    transform = AxesTransform(original_axes, tuple(original_shape))
    current_axes = list(transform.dl_axes)
    sample_dims = transform.sample_dims

    # restore sample dimensions
    if len(sample_dims) > 1:
        # multiple dimensions where multiplexed
        sizes = tuple(transform.dim_sizes[d] for d in sample_dims)
        array = array.reshape(sizes + array.shape[1:])
        current_axes = list(sample_dims) + current_axes[1:]
    elif len(sample_dims) == 1 and sample_dims[0] != "S":
        # relabel sample axis (typically S to T)
        current_axes[0] = sample_dims[0]
    elif len(sample_dims) == 0:
        # remove singleton
        array = np.squeeze(array, axis=0)
        current_axes.pop(0)

    # remove singleton C
    if transform.c_added:
        c_idx = current_axes.index("C")
        array = np.squeeze(array, axis=c_idx)
        current_axes.pop(c_idx)

    # reorder axes to original order
    current_str = "".join(current_axes)
    if current_str != original_axes:
        source = [current_str.index(a) for a in original_axes]
        array = np.moveaxis(array, source, list(range(len(original_axes))))

    return array


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

    transform = AxesTransform(original_axes, tuple(original_shape))

    # remove sample dim
    current_axes = list(transform.dl_axes[1:])

    # remove singleton C if it was added
    if transform.c_added:
        tile = np.squeeze(tile, axis=0)
        current_axes.pop(0)

    # expected dimensions for a tile in original space
    tile_dims = set(transform.dl_axes[1:])
    tile_original = "".join(a for a in original_axes if a in tile_dims)

    current_str = "".join(current_axes)
    if current_str != tile_original:
        source = [current_str.index(a) for a in tile_original]
        tile = np.moveaxis(tile, source, list(range(len(tile_original))))

    return tile


def get_original_stitch_slices(
    array_shape: Sequence[int],
    original_axes: str,
    original_shape: Sequence[int],
    sample_idx: int,
    stitch_coords: Sequence[int],
    crop_size: Sequence[int],
) -> tuple[slice | int, ...]:
    """Get slices to stitch tile back into original array.

    Parameters
    ----------
    array_shape : Sequence[int]
        Shape of the array in DL space (e.g. SCZYX).
    original_axes : str
        Original axes string of the full data.
    original_shape : Sequence[int]
        Original shape of the full data.
    sample_idx : int
        Index of the sample in DL space (S axis) to stitch back.
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
        sample_dim_sizes = [transform.dim_sizes[d] for d in sample_dims]
        sample_indices = [
            int(i) for i in np.unravel_index(sample_idx, sample_dim_sizes)
        ]
        stitch_slices.extend(sample_indices)
        which_axes.extend(sample_dims)

    if not transform.c_added:
        stitch_slices.append(slice(0, array_shape[1]))
        which_axes.append("C")

    # add spatial slices
    stitch_slices.extend(
        [
            slice(start, start + length)
            for start, length in zip(stitch_coords, crop_size, strict=True)
        ]
    )
    which_axes.extend([a for a in transform.dl_axes if a in "ZYX"])
    assert len(stitch_slices) == len(transform.original_axes)

    # reorder slices
    stitch_slices = [
        stitch_slices[which_axes.index(a)] for a in transform.original_axes
    ]

    return tuple(stitch_slices)
