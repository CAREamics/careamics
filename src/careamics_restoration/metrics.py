import numpy as np
from skimage.metrics import peak_signal_noise_ratio


def psnr(gt: np.ndarray, pred: np.ndarray, range: float = 255.0) -> float:
    """Peak Signal to Noise Ratio.

    This method calls skimage.metrics.peak_signal_noise_ratio. See:
    https://scikit-image.org/docs/dev/api/skimage.metrics.html

    Parameters
    ----------
    gt : NumPy array
        Ground truth image
    pred : NumPy array
        Predicted image
    range : float, optional
        The images pixel range, by default 255.0

    Returns
    -------
    float
        PSNR value
    """
    return peak_signal_noise_ratio(gt, pred, data_range=range)


def zero_mean(x: np.ndarray) -> np.ndarray:
    """Zero the mean of an array.

    Parameters
    ----------
    x : NumPy array
        Input array

    Returns
    -------
    NumPy array
        Zero-mean array
    """
    return x - np.mean(x)


def fix_range(gt: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Adjust the range of an array."""
    a = np.sum(gt * x) / (np.sum(x * x))
    return x * a


def fix(gt: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Zero mean the groud truth."""
    gt_ = zero_mean(gt)
    return fix_range(gt_, zero_mean(x))


def scale_invariant_psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    """Scale invariant PSNR.

    Parameters
    ----------
    gt : NumPy array
        Ground truth image
    pred : NumPy array
        Predicted image

    Returns
    -------
    Callable
        Scale invariant PSNR
    """
    range_parameter = (np.max(gt) - np.min(gt)) / np.std(gt)
    gt_ = zero_mean(gt) / np.std(gt)
    return psnr(zero_mean(gt_), fix(gt_, pred), range_parameter)


class MetricTracker:
    """Metric tracker.

    This class is used to track values, sum, count and average of a metric over time.

    Attributes
    ----------
    val : int
        Last value of the metric
    avg : float
        Average value of the metric
    sum : int
        Sum of the metric values (times number of values)
    count : int
        Number of values
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset the metric tracker state."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, value: int, n: int = 1) -> None:
        """Update the metric tracker state.

        Parameters
        ----------
        value : int
            Value to update the metric tracker with
        n : int
            Number of values, equals to batch size
        """
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
