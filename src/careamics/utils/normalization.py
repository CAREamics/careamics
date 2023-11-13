"""
Normalization submodule.

These methods are used to normalize and denormalize images.
"""
from multiprocessing import Value

import numpy as np


def normalize(img: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Normalize an image using mean and standard deviation.

    Images are normalised by subtracting the mean and dividing by the standard
    deviation.

    Parameters
    ----------
    img : np.ndarray
        Image to normalize.
    mean : float
        Mean.
    std : float
        Standard deviation.

    Returns
    -------
    np.ndarray
        Normalized array.
    """
    zero_mean = img - mean
    return zero_mean / std


def denormalize(img: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Denormalize an image using mean and standard deviation.

    Images are denormalised by multiplying by the standard deviation and adding the
    mean.

    Parameters
    ----------
    img : np.ndarray
        Image to denormalize.
    mean : float
        Mean.
    std : float
        Standard deviation.

    Returns
    -------
    np.ndarray
        Denormalized array.
    """
    return img * std + mean

class RunningStats:
    """Calculates running mean and std."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset the running stats."""
        self.sum_mean = Value('d', 0)
        self.count_mean = Value('i', 0)
        self.avg_mean = Value('d', 0)
        self.sum_std = Value('d',0)
        self.count_std = Value('d', 0)
        self.avg_std = Value('d', 0)

    def update_mean(self, value: float, n: int = 1) -> None:
        """Update running mean."""
        self.sum_mean.value += value#* n
        self.count_mean.value += n
        self.avg_mean.value = self.sum_mean.value / self.count_mean.value

    def update_std(self, value: float, n: int = 1) -> None:
        """Update running std."""
        self.sum_std.value += value * n
        self.count_std.value += n
        self.avg_std.value = self.sum_std.value / self.count_std.value
