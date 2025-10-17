from collections.abc import Sequence
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T", bound=np.generic)


def pad_patch(
    coords: Sequence[int],
    patch_size: Sequence[int],
    data_shape: Sequence[int],
    patch_data: NDArray[T],
) -> NDArray[T]:
    patch = np.zeros((data_shape[1], *patch_size), dtype=patch_data.dtype)
    patch_start = np.clip(np.array(coords), 0, None) - np.array(coords)
    patch_end = np.array(coords) + np.array(patch_size)
    patch_end = np.clip(patch_end, None, np.array(data_shape[2:])) - np.clip(
        np.array(coords), None, np.array(data_shape[2:])
    )
    patch[
        (
            slice(None, None, None),
            *tuple(slice(s, t) for s, t in zip(patch_start, patch_end, strict=False)),
        )
    ] = patch_data
    return patch
