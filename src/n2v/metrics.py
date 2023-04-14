import numpy as np
from skimage.metrics import peak_signal_noise_ratio


def psnr(gt, pred, range=255.0):
    return peak_signal_noise_ratio(gt, pred, data_range=range)


def zero_mean(x):
    return x-np.mean(x)


def fix_range(gt,x):
    a = np.sum(gt*x) / (np.sum(x*x))
    return x*a


def fix(gt,x):
    gt_ = zero_mean(gt)
    return fix_range(gt_, zero_mean(x))


def scale_psnr(gt, pred):
    """Scale invariant PSNR

    Parameters
    ----------
    gt : _type_
        _description_
    pred : _type_
        _description_

    Returns
    -------
    Callable
        _description_
    """
    range = (np.max(gt) - np.min(gt)) / np.std(gt)
    gt_ = zero_mean(gt) / np.std(gt)
    return psnr(zero_mean(gt_), fix(gt_, pred), range)


class MetricTracker:

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value: int, n: int = 1):
        """Update the metric tracker state

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