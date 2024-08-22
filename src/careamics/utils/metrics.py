"""
Metrics submodule.

This module contains various metrics and a metrics tracking class.
"""

# NOTE: this doesn't work with torch tensors, since `torch` refuses to
# compute the `mean()` or `std()` of a tensor whose dtype is not float.


from typing import Union

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio

Array = Union[np.ndarray, torch.Tensor]


def psnr(gt: Array, pred: Array, range: float = 255.0) -> float:
    """
    Peak Signal to Noise Ratio.

    This method calls skimage.metrics.peak_signal_noise_ratio. See:
    https://scikit-image.org/docs/dev/api/skimage.metrics.html.

    Parameters
    ----------
    gt : Array
        Ground truth image.
    pred : Array
        Predicted image.
    range : float, optional
        The images pixel range, by default 255.0.

    Returns
    -------
    float
        PSNR value.
    """
    return peak_signal_noise_ratio(gt, pred, data_range=range)


def _zero_mean(x: Array) -> Array:
    """
    Zero the mean of an array.

    Parameters
    ----------
    x : Array
        Input array.

    Returns
    -------
    Array
        Zero-mean array.
    """
    type_ = type(x)
    dtype_ = x.dtype
    x = np.asarray(x)
    res = x - x.mean()
    if type_ == torch.Tensor:
        return torch.tensor(res, dtype=dtype_)
    elif type_ == np.ndarray:
        return res


def _fix_range(gt: Array, x: Array) -> Array:
    """
    Adjust the range of an array based on a reference ground-truth array.

    Parameters
    ----------
    gt : Array
        Ground truth image.
    x : Array
        Input array.

    Returns
    -------
    Array
        Range-adjusted array.
    """
    a = (gt * x).sum() / (x * x).sum()
    return x * a


def _fix(gt: Array, x: Array) -> Array:
    """
    Zero mean a groud truth array and adjust the range of the array.

    Parameters
    ----------
    gt : Array
        Ground truth image.
    x : Array
        Input array.

    Returns
    -------
    Array
        Zero-mean and range-adjusted array.
    """
    gt_ = _zero_mean(gt)
    return _fix_range(gt_, _zero_mean(x))


def scale_invariant_psnr(gt: Array, pred: Array) -> Union[float, torch.tensor]:
    """
    Scale invariant PSNR.

    Parameters
    ----------
    gt : Array
        Ground truth image.
    pred : Array
        Predicted image.

    Returns
    -------
    Union[float, torch.tensor]
        Scale invariant PSNR value.
    """
    range_parameter = (gt.max() - gt.min()) / gt.std()
    gt_ = _zero_mean(gt) / gt.std()
    return psnr(_zero_mean(gt_), _fix(gt_, pred), range_parameter)


class RunningPSNR:
    """Compute the running PSNR during validation step in training.

    This class allows to compute the PSNR on the entire validation set
    one batch at the time.

    Attributes
    ----------
    N : int
        Number of elements seen so far during the epoch.
    mse_sum : float
        Running sum of the MSE over the N elements seen so far.
    max : float
        Running max value of the N target images seen so far.
    min : float
        Running min value of the N target images seen so far.
    """

    def __init__(self):
        self.N = None
        self.mse_sum = None
        self.max = self.min = None
        self.reset()

    def reset(self):
        """Reset the running PSNR computation.

        Usually called at the end of each epoch.
        """
        self.mse_sum = 0
        self.N = 0
        self.max = self.min = None

    def update(self, rec: torch.Tensor, tar: torch.Tensor) -> None:
        """Update the running PSNR statistics given a new batch.

        Parameters
        ----------
        rec: torch.Tensor
            Batch of reconstructed images (B, C, H, W).
        tar: torch.Tensor
            Batch of target images (B, C, H, W).
        """
        ins_max = torch.max(tar).item()
        ins_min = torch.min(tar).item()
        if self.max is None:
            assert self.min is None
            self.max = ins_max
            self.min = ins_min
        else:
            self.max = max(self.max, ins_max)
            self.min = min(self.min, ins_min)

        mse = (rec - tar) ** 2
        elementwise_mse = torch.mean(mse.view(len(mse), -1), dim=1)
        self.mse_sum += torch.nansum(elementwise_mse)
        self.N += len(elementwise_mse) - torch.sum(torch.isnan(elementwise_mse))

    def get(self):
        """Get the actual PSNR value given the running statistics."""
        if self.N == 0 or self.N is None:
            return None
        rmse = torch.sqrt(self.mse_sum / self.N)
        return 20 * torch.log10((self.max - self.min) / rmse)
