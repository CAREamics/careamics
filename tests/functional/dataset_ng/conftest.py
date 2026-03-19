from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from careamics.dataset_ng.patching_strategies import PatchingStrategy


def track_patching(
    patching: PatchingStrategy,
    data_shapes: Sequence[Sequence[int]],
    patch_size: Sequence[int],
    epochs: int,
) -> Sequence[NDArray[np.int_]]:
    """
    Utility function to track where patches are selected from over a number of epochs.
    """
    tracking_arrays = [np.zeros(shape, dtype=int) for shape in data_shapes]
    for _ in range(epochs):
        for i in range(patching.n_patches):
            patch_spec = patching.get_patch_spec(i)
            data_idx = patch_spec["data_idx"]
            sample_idx = patch_spec["sample_idx"]
            coord = patch_spec["coords"]
            patch_slice = [
                slice(c, c + ps) for c, ps in zip(coord, patch_size, strict=True)
            ]
            tracking_arrays[data_idx][sample_idx, ..., *patch_slice] += 1
    return tracking_arrays
