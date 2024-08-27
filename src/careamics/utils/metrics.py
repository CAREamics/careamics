"""
Metrics submodule.

This module contains various metrics and a metrics tracking class.
"""

from typing import Optional, Union

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
# TODO: does this add additional dependency? 


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
        """Constructor."""
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
        rec : torch.Tensor
            Reconstructed batch.
        tar : torch.Tensor
            Target batch.
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

    def get(self) -> Optional[torch.Tensor]:
        """Get the actual PSNR value given the running statistics.

        Returns
        -------
        Optional[torch.Tensor]
            PSNR value.
        """
        if self.N == 0 or self.N is None:
            return None
        rmse = torch.sqrt(self.mse_sum / self.N)
        return 20 * torch.log10((self.max - self.min) / rmse)
    

def _range_invariant_multiscale_ssim(
    gt_: Union[np.ndarray, torch.Tensor], 
    pred_: Union[np.ndarray, torch.Tensor]
):
    """Compute range invariant multiscale SSIM for one channel.
    
    The advantage of this metric in comparison to commonly used SSIM is that
    it is invariant to scalar multiplications in the prediction.
    # TODO: Add reference to the paper.
    
    NOTE: images fed to this function should have channels dimension as the last one.
    
    Parameters:
    ----------
    gt_: Union[np.ndarray, torch.Tensor]
        Ground truth image with shape (N, H, W, C).
    pred_: Union[np.ndarray, torch.Tensor]
        Predicted image with shape (N, H, W, C).
    """
    shape = gt_.shape
    gt_ = torch.Tensor(gt_.reshape((shape[0], -1)))
    pred_ = torch.Tensor(pred_.reshape((shape[0], -1)))
    gt_ = _zero_mean(gt_)
    pred_ = _zero_mean(pred_)
    pred_ = _fix(gt_, pred_)
    pred_ = pred_.reshape(shape)
    gt_ = gt_.reshape(shape)

    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(
        data_range=gt_.max() - gt_.min()
    )
    return ms_ssim(torch.Tensor(pred_[:, None]), torch.Tensor(gt_[:, None])).item()


def multiscale_ssim(
    gt_: Union[np.ndarray, torch.Tensor], 
    pred_: Union[np.ndarray, torch.Tensor],
    range_invariant: bool = True
):
    """Compute channel-wise multiscale SSIM for each channel.
    
    It allows to use either standard multiscale SSIM or its range-invariant version.
    
    NOTE: images fed to this function should have channels dimension as the last one.
    # TODO: do we want to allow this behavior? or we want the usual (N, C, H, W)?
    
    Parameters:
    ----------
    gt_: Union[np.ndarray, torch.Tensor]
        Ground truth image with shape (N, H, W, C).
    pred_: Union[np.ndarray, torch.Tensor]
        Predicted image with shape (N, H, W, C).
    range_invariant: bool 
        Whether to use standard or range invariant multiscale SSIM.
    """
    ms_ssim_values = {i: None for i in range(gt_.shape[-1])}
    for ch_idx in range(gt_.shape[-1]):
        tar_tmp = gt_[..., ch_idx]
        pred_tmp = pred_[..., ch_idx]
        if range_invariant:
            ms_ssim_values[ch_idx] = _range_invariant_multiscale_ssim(
                gt_=tar_tmp, pred_=pred_tmp
            )
        else:
            ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(
                data_range=tar_tmp.max() - tar_tmp.min()
            )
            ms_ssim_values[ch_idx] = ms_ssim(
                torch.Tensor(pred_tmp[:, None]), torch.Tensor(tar_tmp[:, None])
            ).item()

    output = [ms_ssim_values[i] for i in range(gt_.shape[-1])]
    return output
