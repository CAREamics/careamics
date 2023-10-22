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

class RunningStats:
    """Calculates running mean and std."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset the running stats."""
        self.sum_mean = 0
        self.count_mean = 0
        self.avg_mean = 0
        self.sum_std = 0
        self.count_std = 0
        self.avg_std = 0

    def update_mean(self, value: float, n: int = 1) -> None:
        """Update running mean."""
        self.sum_mean += value * n
        self.count_mean += n
        self.avg_mean = self.sum_mean / self.count_mean

    def update_std(self, value: float, n: int = 1) -> None:
        """Update running std."""
        self.sum_std += value * n
        self.count_std += n
        self.avg_std = self.sum_std / self.count_std
