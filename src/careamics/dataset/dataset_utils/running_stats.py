"""Computing data statistics."""

import numpy as np
from numpy.typing import NDArray


def compute_normalization_stats(image: NDArray) -> tuple[NDArray, NDArray]:
    """
    Compute mean and standard deviation of an array.

    Expected input shape is (S, C, (Z), Y, X). The mean and standard deviation are
    computed per channel.

    Parameters
    ----------
    image : NDArray
        Input array.

    Returns
    -------
    tuple of (list of floats, list of floats)
        Lists of mean and standard deviation values per channel.
    """
    # Define the list of axes excluding the channel axis
    axes = tuple(np.delete(np.arange(image.ndim), 1))
    return np.mean(image, axis=axes), np.std(image, axis=axes)


def update_iterative_stats(
    count: NDArray, mean: NDArray, m2: NDArray, new_values: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    """Update the mean and variance of an array iteratively.

    Parameters
    ----------
    count : NDArray
        Number of elements in the array.
    mean : NDArray
        Mean of the array.
    m2 : NDArray
        Variance of the array.
    new_values : NDArray
        New values to add to the mean and variance.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray]
        Updated count, mean, and variance.
    """
    count += np.array([np.prod(channel.shape) for channel in new_values])
    # newvalues - oldMean
    delta = [
        np.subtract(v.flatten(), [m] * len(v.flatten()))
        for v, m in zip(new_values, mean)
    ]

    mean += np.array([np.sum(d / c) for d, c in zip(delta, count)])
    # newvalues - newMeant
    delta2 = [
        np.subtract(v.flatten(), [m] * len(v.flatten()))
        for v, m in zip(new_values, mean)
    ]

    m2 += np.array([np.sum(d * d2) for d, d2 in zip(delta, delta2)])

    return (count, mean, m2)


def finalize_iterative_stats(
    count: NDArray, mean: NDArray, m2: NDArray
) -> tuple[NDArray, NDArray]:
    """Finalize the mean and variance computation.

    Parameters
    ----------
    count : NDArray
        Number of elements in the array.
    mean : NDArray
        Mean of the array.
    m2 : NDArray
        Variance of the array.

    Returns
    -------
    tuple[NDArray, NDArray]
        Final mean and standard deviation.
    """
    std = np.array([np.sqrt(m / c) for m, c in zip(m2, count)])
    if any(c < 2 for c in count):
        return np.full(mean.shape, np.nan), np.full(std.shape, np.nan)
    else:
        return mean, std


class WelfordStatistics:
    """Compute Welford statistics iteratively.

    The Welford algorithm is used to compute the mean and variance of an array
    iteratively. Based on the implementation from:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """

    def update(self, array: NDArray, sample_idx: int) -> None:
        """Update the Welford statistics.

        Parameters
        ----------
        array : NDArray
            Input array.
        sample_idx : int
            Current sample number.
        """
        self.sample_idx = sample_idx
        sample_channels = np.array(np.split(array, array.shape[1], axis=1))

        # Initialize the statistics
        if self.sample_idx == 0:
            # Compute the mean and standard deviation
            self.mean, _ = compute_normalization_stats(array)
            # Initialize the count and m2 with zero-valued arrays of shape (C,)
            self.count, self.mean, self.m2 = update_iterative_stats(
                count=np.zeros(array.shape[1]),
                mean=self.mean,
                m2=np.zeros(array.shape[1]),
                new_values=sample_channels,
            )
        else:
            # Update the statistics
            self.count, self.mean, self.m2 = update_iterative_stats(
                count=self.count, mean=self.mean, m2=self.m2, new_values=sample_channels
            )

        self.sample_idx += 1

    def finalize(self) -> tuple[NDArray, NDArray]:
        """Finalize the Welford statistics.

        Returns
        -------
        tuple or numpy arrays
            Final mean and standard deviation.
        """
        return finalize_iterative_stats(self.count, self.mean, self.m2)


# from multiprocessing import Value
# from typing import tuple

# import numpy as np


# class RunningStats:
#     """Calculates running mean and std."""

#     def __init__(self) -> None:
#         self.reset()

#     def reset(self) -> None:
#         """Reset the running stats."""
#         self.avg_mean = Value("d", 0)
#         self.avg_std = Value("d", 0)
#         self.m2 = Value("d", 0)
#         self.count = Value("i", 0)

#     def init(self, mean: float, std: float) -> None:
#         """Initialize running stats."""
#         with self.avg_mean.get_lock():
#             self.avg_mean.value += mean
#         with self.avg_std.get_lock():
#             self.avg_std.value = std

#     def compute_std(self) -> tuple[float, float]:
#         """Compute std."""
#         if self.count.value >= 2:
#             self.avg_std.value = np.sqrt(self.m2.value / self.count.value)

#     def update(self, value: float) -> None:
#         """Update running stats."""
#         with self.count.get_lock():
#             self.count.value += 1
#         delta = value - self.avg_mean.value
#         with self.avg_mean.get_lock():
#             self.avg_mean.value += delta / self.count.value
#         delta2 = value - self.avg_mean.value
#         with self.m2.get_lock():
#             self.m2.value += delta * delta2
