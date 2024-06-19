"""
This script contains the functions/classes to compute loss and metrics used to train and evaluate the performance of the model.
"""

import numpy as np
import torch
from skimage.metrics import structural_similarity
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

from careamics.models.lvae.utils import allow_numpy


class RunningPSNR:
    """
    This class allows to compute the running PSNR during validation step in training.
    In this way it is possible to compute the PSNR on the entire validation set one batch at the time.
    """

    def __init__(self):
        # number of elements seen so far during the epoch
        self.N = None
        # running sum of the MSE over the self.N elements seen so far
        self.mse_sum = None
        # running max and min values of the self.N target images seen so far
        self.max = self.min = None
        self.reset()

    def reset(self):
        """
        Used to reset the running PSNR (usually called at the end of each epoch).
        """
        self.mse_sum = 0
        self.N = 0
        self.max = self.min = None

    def update(self, rec: torch.Tensor, tar: torch.Tensor) -> None:
        """
        Given a batch of reconstructed and target images, it updates the MSE and.

        Parameters
        ----------
        rec: torch.Tensor
            Batch of reconstructed images (B, H, W).
        tar: torch.Tensor
            Batch of target images (B, H, W).
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
        """
        The get the actual PSNR value given the running statistics.
        """
        if self.N == 0 or self.N is None:
            return None
        rmse = torch.sqrt(self.mse_sum / self.N)
        return 20 * torch.log10((self.max - self.min) / rmse)


def zero_mean(x):
    return x - torch.mean(x, dim=1, keepdim=True)


def fix_range(gt, x):
    a = torch.sum(gt * x, dim=1, keepdim=True) / (torch.sum(x * x, dim=1, keepdim=True))
    return x * a


def fix(gt, x):
    gt_ = zero_mean(gt)
    return fix_range(gt_, zero_mean(x))


def _PSNR_internal(gt, pred, range_=None):
    if range_ is None:
        range_ = torch.max(gt, dim=1).values - torch.min(gt, dim=1).values

    mse = torch.mean((gt - pred) ** 2, dim=1)
    return 20 * torch.log10(range_ / torch.sqrt(mse))


@allow_numpy
def PSNR(gt, pred, range_=None):
    """
    Compute PSNR.

    Parameters
    ----------
    gt: array
        Ground truth image.
    pred: array
        Predicted image.
    """
    assert len(gt.shape) == 3, "Images must be in shape: (batch,H,W)"

    gt = gt.view(len(gt), -1)
    pred = pred.view(len(gt), -1)
    return _PSNR_internal(gt, pred, range_=range_)


@allow_numpy
def RangeInvariantPsnr(gt: torch.Tensor, pred: torch.Tensor):
    """
    NOTE: Works only for grayscale images.
    Adapted from https://github.com/juglab/ScaleInvPSNR/blob/master/psnr.py
    It rescales the prediction to ensure that the prediction has the same range as the ground truth.
    """
    assert len(gt.shape) == 3, "Images must be in shape: (batch,H,W)"
    gt = gt.view(len(gt), -1)
    pred = pred.view(len(gt), -1)
    ra = (torch.max(gt, dim=1).values - torch.min(gt, dim=1).values) / torch.std(
        gt, dim=1
    )
    gt_ = zero_mean(gt) / torch.std(gt, dim=1, keepdim=True)
    return _PSNR_internal(zero_mean(gt_), fix(gt_, pred), ra)


def _avg_psnr(target, prediction, psnr_fn):
    output = np.mean(
        [
            psnr_fn(target[i : i + 1], prediction[i : i + 1]).item()
            for i in range(len(prediction))
        ]
    )
    return round(output, 2)


def avg_range_inv_psnr(target, prediction):
    return _avg_psnr(target, prediction, RangeInvariantPsnr)


def avg_psnr(target, prediction):
    return _avg_psnr(target, prediction, PSNR)


def compute_masked_psnr(mask, tar1, tar2, pred1, pred2):
    mask = mask.astype(bool)
    mask = mask[..., 0]
    tmp_tar1 = tar1[mask].reshape((len(tar1), -1, 1))
    tmp_pred1 = pred1[mask].reshape((len(tar1), -1, 1))
    tmp_tar2 = tar2[mask].reshape((len(tar2), -1, 1))
    tmp_pred2 = pred2[mask].reshape((len(tar2), -1, 1))
    psnr1 = avg_range_inv_psnr(tmp_tar1, tmp_pred1)
    psnr2 = avg_range_inv_psnr(tmp_tar2, tmp_pred2)
    return psnr1, psnr2


def avg_ssim(target, prediction):
    ssim = [
        structural_similarity(
            target[i], prediction[i], data_range=(target[i].max() - target[i].min())
        )
        for i in range(len(target))
    ]
    return np.mean(ssim), np.std(ssim)


@allow_numpy
def range_invariant_multiscale_ssim(gt_, pred_):
    """
    Computes range invariant multiscale ssim for one channel.
    This has the benefit that it is invariant to scalar multiplications in the prediction.
    """
    shape = gt_.shape
    gt_ = torch.Tensor(gt_.reshape((shape[0], -1)))
    pred_ = torch.Tensor(pred_.reshape((shape[0], -1)))
    gt_ = zero_mean(gt_)
    pred_ = zero_mean(pred_)
    pred_ = fix(gt_, pred_)
    pred_ = pred_.reshape(shape)
    gt_ = gt_.reshape(shape)

    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(
        data_range=gt_.max() - gt_.min()
    )
    return ms_ssim(torch.Tensor(pred_[:, None]), torch.Tensor(gt_[:, None])).item()


def compute_multiscale_ssim(gt_, pred_, range_invariant=True):
    """
    Computes multiscale ssim for each channel.
    Args:
    gt_: ground truth image with shape (N, H, W, C)
    pred_: predicted image with shape (N, H, W, C)
    range_invariant: whether to use range invariant multiscale ssim
    """
    ms_ssim_values = {i: None for i in range(gt_.shape[-1])}
    for ch_idx in range(gt_.shape[-1]):
        tar_tmp = gt_[..., ch_idx]
        pred_tmp = pred_[..., ch_idx]
        if range_invariant:
            ms_ssim_values[ch_idx] = range_invariant_multiscale_ssim(tar_tmp, pred_tmp)
        else:
            ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(
                data_range=tar_tmp.max() - tar_tmp.min()
            )
            ms_ssim_values[ch_idx] = ms_ssim(
                torch.Tensor(pred_tmp[:, None]), torch.Tensor(tar_tmp[:, None])
            ).item()

    output = [ms_ssim_values[i] for i in range(gt_.shape[-1])]
    return output
