"""
This script contains the functions/classes to compute loss and metrics used to train and evaluate the performance of the model.
"""
import torch

from .utils import allow_numpy

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

    def update(
        self, 
        rec: torch.Tensor, 
        tar: torch.Tensor
    ) -> None:
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

        mse = (rec - tar)**2
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
def RangeInvariantPsnr(
    gt: torch.Tensor, 
    pred: torch.Tensor
):
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

class MetricMonitor:
    def __init__(self, metric):
        assert metric in ['val_loss', 'val_psnr']
        self.metric = metric

    def mode(self):
        if self.metric == 'val_loss':
            return 'min'
        elif self.metric == 'val_psnr':
            return 'max'
        else:
            raise ValueError(f'Invalid metric:{self.metric}')