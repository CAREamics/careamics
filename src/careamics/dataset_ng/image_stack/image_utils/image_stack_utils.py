from collections.abc import Sequence
from types import EllipsisType
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T", bound=np.generic)


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


# TODO: add tests
# TODO: move to dataset_utils, better name?
def reshape_array_shape(
    original_axes: str, shape: Sequence[int], add_singleton: bool = True
) -> tuple[int, ...]:
    """Find resulting shape if reshaping array to SC(Z)YX.

    If `T` is present in the original axes, its size is multiplied into `S`, as both
    axes are multiplexed.

    Setting `add_singleton` to `False` will only include axes that are present in
    `original_axes` in the output shape.

    Parameters
    ----------
    original_axes : str
        The axes of the original array, e.g. "TCZYX", "SCYX", etc.
    shape : Sequence[int]
        The shape of the original array.
    add_singleton : bool, default=True
        Whether to add singleton dimensions for missing axes. When `False`, only axes
        present in `original_axes` will be included in the output shape. When `True`,
        missing mandatory axes (`S` and `C`) will be added as singleton dimensions.
    """
    target_axes = "SCZYX"
    target_shape = []
    for d in target_axes:
        if d in original_axes:
            idx = original_axes.index(d)
            target_shape.append(shape[idx])
        elif d != "Z":
            if add_singleton:
                target_shape.append(1)

    if "T" in original_axes:
        idx = original_axes.index("T")
        if "S" in original_axes or add_singleton:
            target_shape[0] = target_shape[0] * shape[idx]
        else:
            target_shape.insert(0, shape[idx])

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
    patch = np.zeros((patch_data.shape[0], *patch_size), dtype=patch_data.dtype)
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
