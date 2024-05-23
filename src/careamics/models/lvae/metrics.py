"""
This script contains the functions/classes to compute loss and metrics used to train and evaluate the performance of the model.
"""
from typing import Tuple

import torch
from torch.distributions.normal import Normal

from .utils import (
    StableMean, StableLogVar, 
    allow_numpy
)

def kl_normal_mc(
    z: torch.Tensor, 
    p_mulv: Tuple[StableMean, StableLogVar], 
    q_mulv: Tuple[StableMean, StableLogVar]
) -> torch.Tensor:
    """
    One-sample estimation of element-wise KL between two diagonal multivariate normal distributions.
    Any number of dimensions, broadcasting supported (be careful).
    
    Parameters
    ----------
    z: torch.Tensor
        The sampled latent tensor.
    p_mulv: Tuple[StableMean, StableLogVar]
        A tuple containing the mean and log-variance of the prior generative distribution p(z).
    q_mulv: Tuple[StableMean, StableLogVar]
        A tuple containing the mean and log-variance of the inference distribution q(z).
    """
    assert isinstance(p_mulv, tuple)
    assert isinstance(q_mulv, tuple)
    p_mu, p_lv = p_mulv
    q_mu, q_lv = q_mulv

    p_std = p_lv.get_std()
    q_std = q_lv.get_std()

    p_distrib = Normal(p_mu.get(), p_std)
    q_distrib = Normal(q_mu.get(), q_std)
    return q_distrib.log_prob(z) - p_distrib.log_prob(z)

class RunningPSNR:

    def __init__(self):
        self.N = self.mse_sum = self.max = self.min = None
        self.reset()

    def reset(self):
        self.mse_sum = 0
        self.N = 0
        self.max = self.min = None

    def update(self, rec, tar):
        ins_max = torch.max(tar).item()
        ins_min = torch.min(tar).item()
        if self.max is None:
            assert self.min is None
            self.max = ins_max
            self.min = ins_min
        else:
            self.max = max(self.max, ins_max)
            self.min = min(self.min, ins_min)

        mse = (rec - tar)**2
        elementwise_mse = torch.mean(mse.view(len(mse), -1), dim=1)
        self.mse_sum += torch.nansum(elementwise_mse)
        self.N += len(elementwise_mse) - torch.sum(torch.isnan(elementwise_mse))

    def get(self):
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
    '''
        Compute PSNR.
        Parameters
        ----------
        gt: array
            Ground truth image.
        pred: array
            Predicted image.
    '''
    assert len(gt.shape) == 3, 'Images must be in shape: (batch,H,W)'

    gt = gt.view(len(gt), -1)
    pred = pred.view(len(gt), -1)
    return _PSNR_internal(gt, pred, range_=range_)


@allow_numpy
def RangeInvariantPsnr(gt, pred):
    """
    NOTE: Works only for grayscale images.
    Adapted from https://github.com/juglab/ScaleInvPSNR/blob/master/psnr.py
    It rescales the prediction to ensure that the prediction has the same range as the ground truth.
    """
    assert len(gt.shape) == 3, 'Images must be in shape: (batch,H,W)'
    gt = gt.view(len(gt), -1)
    pred = pred.view(len(gt), -1)
    ra = (torch.max(gt, dim=1).values - torch.min(gt, dim=1).values) / torch.std(gt, dim=1)
    gt_ = zero_mean(gt) / torch.std(gt, dim=1, keepdim=True)
    return _PSNR_internal(zero_mean(gt_), fix(gt_, pred), ra)

