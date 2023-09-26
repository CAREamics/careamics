import numpy as np


def normalize(img: np.ndarray, mean: float, std: float) -> np.ndarray:
    """_summary_.

    _extended_summary_

    Parameters
    ----------
    img : np.ndarray
        _description_
    mean : float
        _description_
    std : float
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    zero_mean = img - mean
    return zero_mean / std


def denormalize(img: np.ndarray, mean: float, std: float) -> np.ndarray:
    """_summary_.

    _extended_summary_

    Parameters
    ----------
    img : np.ndarray
        _description_
    mean : float
        _description_
    std : float
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    return img * std + mean
