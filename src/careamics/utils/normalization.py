"""
Normalization submodule.

These methods are used to normalize and denormalize images.
"""
from typing import List, Tuple, Union

import numpy as np


def normalize(
    img: Union[Tuple, np.ndarray], mean: float, std: float
) -> Union[List, np.ndarray]:
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
