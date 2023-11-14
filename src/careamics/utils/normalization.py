"""
Normalization submodule.

These methods are used to normalize and denormalize images.
"""
from multiprocessing import Value
from typing import Tuple

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
        self.avg_mean = Value('d', 0)
        self.avg_std = Value('d', 0)
        self.m2 = Value('d', 0)
        self.count = Value('i', 0)

    def init(self, mean: float, std: float) -> None:
        """Initialize running stats."""
        with self.avg_mean.get_lock():
            self.avg_mean.value += mean
        with self.avg_std.get_lock():
            self.avg_std.value = std

    def compute_std(self) -> Tuple[float, float]:
        """Compute mean and std."""
        if self.count.value >= 2:
            self.avg_std.value = np.sqrt(self.m2.value / self.count.value)

    def update(self, value: float) -> None:
        """Update running std."""
        with self.count.get_lock():
            self.count.value += 1
        delta = value - self.avg_mean.value
        with self.avg_mean.get_lock():
            self.avg_mean.value += delta / self.count.value
        delta2 = value - self.avg_mean.value
        with self.m2.get_lock():
            self.m2.value += delta * delta2
