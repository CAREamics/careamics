"""Histogram utilities for noise-model training."""

import numpy as np
from numpy.typing import NDArray


def create_histogram(
    bins: int, min_val: float, max_val: float, observation: NDArray, signal: NDArray
) -> NDArray:
    """Create a normalized 2D histogram relating observations to signals.

    Parameters
    ----------
    bins : int
        Number of bins along each histogram axis.
    min_val : float
        Lower bound of the histogram range for both signal and observation values.
    max_val : float
        Upper bound of the histogram range for both signal and observation values.
    observation : numpy.ndarray
        Stack of observed images. Its first dimension must be an integer multiple
        of the first dimension of `signal`.
    signal : numpy.ndarray
        Stack of corresponding signal images. Each signal image is paired with an
        equal number of consecutive observations.

    Returns
    -------
    numpy.ndarray
        Array with shape `(3, bins, bins)`. The first plane contains normalized
        histogram counts; the second and third planes contain the lower and upper
        bin boundaries along the signal axis.
    """
    histogram = np.zeros((3, bins, bins))
    value_range = [min_val, max_val]

    obs_to_signal_shape_factor = int(observation.shape[0] / signal.shape[0])
    signal_indices = np.arange(observation.shape[0]) // obs_to_signal_shape_factor
    signal_values = signal[signal_indices].ravel()
    observation_values = observation.ravel()

    count_histogram, signal_edges, _ = np.histogram2d(
        signal_values, observation_values, bins=bins, range=[value_range, value_range]
    )
    row_sums = count_histogram.sum(axis=1, keepdims=True)
    count_histogram /= np.clip(row_sums, a_min=1e-20, a_max=None)

    histogram[0] = count_histogram
    histogram[1] = signal_edges[:-1][..., np.newaxis]
    histogram[2] = signal_edges[1:][..., np.newaxis]
    return histogram
