from collections.abc import Sequence
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T", bound=np.generic)


# TODO: move to dataset_utils, better name?
def reshaped_array_shape(axes: str, shape: Sequence[int]) -> tuple[int, ...]:
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


def pad_patch(
    coords: Sequence[int],
    patch_size: Sequence[int],
    data_shape: Sequence[int],
    patch_data: NDArray[T],
) -> NDArray[T]:
    """
    Pad patch data with zeros where it is outside the bounds of it's source image.

    This ensures the patch data is contained in an array with the expected patch size.

    If `coords` are negative, the start of the patch will be padded with zeros up until
    where the start of the image would be, and this is where the patch data starts.

    If the `coords + patch_size` are greater than the bounds of the image then the
    end of the patch will be filled with zeros.

    Parameters
    ----------
    coords : Sequence[int]
        The coordinates that describe where the patch starts in the spatial dimension of
        the image
    patch_size : Sequence[int]
        The size of the patch in the spatial dimensions.
    data_shape : Sequence[int]
        The shape of the image the patch originates from, must be in the format SC(Z)YX.
    patch_data : NDArray[T]
        The patch data to be padded.

    Returns
    -------
    NDArray[T]
        The resulting padded patch.
    """
    coords_ = np.array(coords)
    patch = np.zeros((data_shape[1], *patch_size), dtype=patch_data.dtype)
    # data start will be zero unless coords are negative
    data_start = np.clip(coords_, 0, None) - coords_
    data_end = data_start + np.array(patch_data.shape[1:])
    patch[
        (
            slice(None, None, None),  # channel slice
            *tuple(slice(s, t) for s, t in zip(data_start, data_end, strict=False)),
        )
    ] = patch_data
    return patch
