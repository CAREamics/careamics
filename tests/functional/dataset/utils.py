from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from careamics.dataset.patching_strategies import PatchingStrategy


def assert_expected_pixel_probability(
    patching: PatchingStrategy,
    data_shapes: Sequence[Sequence[int]],
    patch_size: Sequence[int],
    mean_expected_prob: dict[tuple[int, int], float],
    epochs: int = 100,
):
    """Assert that each sample has the correct mean-expected-pixel-probability.

    The expected-pixel-probability is the probability a pixel will be selected per epoch
    and the mean is calculated over a sample.

    The `mean_expected_prob` is passed a dictionary where the the keys are
    `(data_idx, sample_idx)` pairs. If a value is not set, a probability of 1.0 is
    assumed.
    """
    tracking_arrays = track_patching(patching, data_shapes, patch_size, epochs)
    # assert first sample has lower prob
    for data_idx, tracking_array in enumerate(tracking_arrays):
        for sample_idx, sample_tracking_array in enumerate(tracking_array):
            mean_pixel_prob = np.mean(sample_tracking_array) / epochs
            expected = mean_expected_prob.get((data_idx, sample_idx), 1.0)
            # NOTE: probability depends on n_patches which is calculated as the ceil
            # means the resultant probability won't be exactly that set.
            np.testing.assert_allclose(mean_pixel_prob, expected, rtol=0.1)


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
