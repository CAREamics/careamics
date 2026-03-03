"""Utility to restore predictions to original shape and dimension order."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def restore_original_shape(
    array: NDArray,
    original_axes: str,
    original_data_shape: Sequence[int],
) -> NDArray:
    """
    Restore array to original shape and dimension order.

    Parameters
    ----------
    array : numpy.ndarray
        Array in SC(Z)YX format.
    original_axes : str
        Original axes order of the data.
    original_data_shape : Sequence[int]
        Original shape of the data.

    Returns
    -------
    numpy.ndarray
        Array reshaped to match axes and original_data_shape.
    """
    if len(array.shape) not in (4, 5):
        raise ValueError(
            f"Expected array with 4 or 5 dimensions (SC(Z)YX), got {len(array.shape)}."
        )

    # current axes from array shape (S and C always present)
    current_axes = "SCZYX" if len(array.shape) == 5 else "SCYX"

    # handle special CZI case where T is used as Z
    if "T" in original_axes and "Z" not in original_axes and len(array.shape) == 5:
        original_axes = original_axes.replace("T", "Z")

    # unflatten S dimension
    merged_dims = [dim for dim in original_axes if dim not in current_axes]

    if merged_dims:
        unflattened_sizes = []
        unflattened_dims = []

        if "S" in original_axes:
            s_size = original_data_shape[original_axes.index("S")]
            unflattened_sizes.append(s_size)
            unflattened_dims.append("S")

        for dim in merged_dims:
            dim_size = original_data_shape[original_axes.index(dim)]
            unflattened_sizes.append(dim_size)
            unflattened_dims.append(dim)

        # replace S dimension with unflattened dimensions
        s_idx = current_axes.index("S")  # TODO always 0
        new_shape = list(array.shape)
        new_shape[s_idx : s_idx + 1] = unflattened_sizes
        array = array.reshape(new_shape)

        # update current axes
        current_axes = (
            current_axes[:s_idx] + "".join(unflattened_dims) + current_axes[s_idx + 1 :]
        )

    # remove singleton C if not in original axes
    if "C" not in original_axes:
        c_idx = current_axes.index("C")
        if array.shape[c_idx] == 1:
            array = np.squeeze(array, axis=c_idx)
            current_axes = current_axes[:c_idx] + current_axes[c_idx + 1 :]

    # same for singleton S
    if "S" in current_axes and "S" not in original_axes:
        s_idx = current_axes.index("S")
        if array.shape[s_idx] == 1:
            array = np.squeeze(array, axis=s_idx)
            current_axes = current_axes[:s_idx] + current_axes[s_idx + 1 :]

    # reorder to match original axes
    if current_axes != original_axes:
        source_order = [current_axes.index(axis) for axis in original_axes]
        target_order = list(range(len(original_axes)))
        array = np.moveaxis(array, source_order, target_order)

    return array
