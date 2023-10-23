"""
Metrics submodule.

This module contains various metrics and a metrics tracking class.
"""
from typing import Union

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio


def psnr(gt: np.ndarray, pred: np.ndarray, range: float = 255.0) -> float:
    """
    Peak Signal to Noise Ratio.

    This method calls skimage.metrics.peak_signal_noise_ratio. See:
    https://scikit-image.org/docs/dev/api/skimage.metrics.html.

    Parameters
    ----------
    gt : NumPy array
        Ground truth image.
    pred : NumPy array
        Predicted image.
    range : float, optional
        The images pixel range, by default 255.0.

    Returns
    -------
    float
        PSNR value.
    """
    return peak_signal_noise_ratio(gt, pred, data_range=range)


def _zero_mean(x: np.ndarray) -> np.ndarray:
    """
    Zero the mean of an array.

    Parameters
    ----------
    x : NumPy array
        Input array.

    Returns
    -------
    NumPy array
        Zero-mean array.
    """
    return x - np.mean(x)


def _fix_range(gt: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Adjust the range of an array based on a reference ground-truth array.

    Parameters
    ----------
    gt : np.ndarray
        Ground truth image.
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Range-adjusted array.
    """
    a = np.sum(gt * x) / (np.sum(x * x))
    return x * a


def _fix(gt: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Zero mean a groud truth array and adjust the range of the array.

    Parameters
    ----------
    gt : np.ndarray
        Ground truth image.
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Zero-mean and range-adjusted array.
    """
    gt_ = _zero_mean(gt)
    return _fix_range(gt_, _zero_mean(x))


def scale_invariant_psnr(
    gt: np.ndarray, pred: np.ndarray
) -> Union[float, torch.tensor]:
    """
    Scale invariant PSNR.

    Parameters
    ----------
    gt : np.ndarray
        Ground truth image.
    pred : np.ndarray
        Predicted image.

    Returns
    -------
    Union[float, torch.tensor]
        Scale invariant PSNR value.
    """
    range_parameter = (np.max(gt) - np.min(gt)) / np.std(gt)
    gt_ = _zero_mean(gt) / np.std(gt)
    return psnr(_zero_mean(gt_), _fix(gt_, pred), range_parameter)


class MetricTracker:
    """
    Metric tracker class.

    This class is used to track values, sum, count and average of a metric over time.

    Attributes
    ----------
    val : int
        Last value of the metric.
    avg : torch.Tensor.float
        Average value of the metric.
    sum : int
        Sum of the metric values (times number of values).
    count : int
        Number of values.
    """

    def __init__(self) -> None:
        """Constructor."""
        self.reset()

    def reset(self) -> None:
        """Reset the metric tracker state."""
        self.val = 0.0
        self.avg: torch.Tensor.float = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, value: int, n: int = 1) -> None:
        """
        Update the metric tracker state.

        Parameters
        ----------
        value : int
            Value to update the metric tracker with.
        n : int
            Number of values, equals to batch size.
        """
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
