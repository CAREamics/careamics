"""PSNR metrics compatible with torchmetrics."""

from typing import Any

import torch
from torch import Tensor, tensor
from torchmetrics import Metric

# Note: LVAE use a PSNR implementation that is averaging over pixels by excluding NaNs,
# and counting the number of non-NaN pixels for averaging. Current choice is to ignore
# that implementation since NaNs should not be present in the model, being a sign of a
# more fundamental issue with the training.

# Note: Running PSNR, and probably SI-PSNR even more, can lead to very different results
# between patches and whole images, so users should be warned.

# TODO add plot function? see torchmetrics docs


def normalise_range(gt: Tensor, pred: Tensor) -> tuple[Tensor, Tensor]:
    """Normalize the range of ground truth and prediction tensors.

    Parameters
    ----------
    gt : Tensor
        Ground truth tensor.
    pred : Tensor
        Prediction tensor.

    Returns
    -------
    torch.Tensor
        Rescaled ground truth tensor.
    torch.Tensor
        Rescaled prediction tensor.
    """
    eps = torch.finfo(gt.dtype).eps

    if len(gt.shape) < 4 or len(gt.shape) > 5:
        raise ValueError(
            f"Input tensors must have 4 or 5 dimensions (B, C, (Z), Y, X), got "
            f"shape {tuple(gt.shape)}."
        )

    dims = tuple(range(2, len(gt.shape)))

    # normalize range of gt and prediction
    gt_rescaled = gt - torch.mean(gt, dim=dims, keepdim=True)
    pred_zero_mean = pred - torch.mean(pred, dim=dims, keepdim=True)

    # scale prediction
    alpha_num = torch.sum(gt_rescaled * pred_zero_mean, dim=dims, keepdim=True) + eps
    alpha_den = torch.sum(pred_zero_mean * pred_zero_mean, dim=dims, keepdim=True) + eps
    alpha = alpha_num / alpha_den

    pred_rescaled = pred_zero_mean * alpha

    return gt_rescaled, pred_rescaled


class GlobSIPSNR(Metric):
    """Scale Invariant PSNR metric with global data range.

    Adapted from juglab/ScaleInvPSNR, this version of PSNR rescales the predictions and
    ground truth to have similar range, then computes the PSNR using a global data range
    accumulated over all batches.

    Parameters
    ----------
    n_channels : int
        Number of channels in the input images.
    **kwargs : Any
        Additional keyword arguments passed to the parent Metric class.

    Attributes
    ----------
    glob_max : Tensor
        Global maximum values for each channel.
    glob_min : Tensor
        Global minimum values for each channel.
    mse_log : Tensor
        Logarithm of the mean squared error summed over batches.
    total : Tensor
        Total number of samples processed.
    """

    def __init__(self, n_channels: int, **kwargs: Any):
        """Initialize the scale invariant PSNR metric.

        Parameters
        ----------
        n_channels : int
            Number of channels in the input images.
        **kwargs : Any
            Additional keyword arguments passed to the parent Metric class.
        """
        super().__init__(**kwargs)

        self.eps = torch.finfo(torch.float32).eps

        self.add_state(
            "glob_max",
            default=tensor([float("-inf") for _ in range(n_channels)]),
            dist_reduce_fx="max",
        )
        self.add_state(
            "glob_min",
            default=tensor([float("inf") for _ in range(n_channels)]),
            dist_reduce_fx="min",
        )
        self.add_state(
            "mse_log",
            default=tensor([0.0 for _ in range(n_channels)]),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total",
            default=tensor([0.0 for _ in range(n_channels)]),
            dist_reduce_fx="sum",
        )

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update the metric states with values computed from a new batch.

        Parameters
        ----------
        preds : Tensor
            Predicted images tensor of shape (B, C, (Z), Y, X).
        target : Tensor
            Ground truth images tensor of shape (B, C, (Z), Y, X).
        """
        batch_size = target.shape[0]
        shape = target.shape
        dims = tuple(range(2, len(shape)))

        # compute min/max of the batches and channels
        batch_min = torch.amin(target, dim=(0,) + dims)
        batch_max = torch.amax(target, dim=(0,) + dims)

        # fix range of gt and prediction
        tar_rescaled, pred_rescaled = normalise_range(target, preds)

        # compute mse
        mse = torch.mean((tar_rescaled - pred_rescaled) ** 2 + self.eps, dim=dims)

        # update states
        self.glob_max: torch.Tensor = torch.maximum(self.glob_max, batch_max)
        self.glob_min: torch.Tensor = torch.minimum(self.glob_min, batch_min)
        self.mse_log += torch.log10(mse).sum(dim=0)
        self.total += batch_size

    def compute(self) -> Tensor | dict[str, Tensor]:
        """Compute the final metric value.

        Returns
        -------
        torch.Tensor
            Tensor of length C containing the computed PSNR for each channel.
        """
        glob_data_range = self.glob_max - self.glob_min + self.eps
        return 10 * (torch.log10(glob_data_range**2) - self.mse_log / self.total)


class LocalSIPSNR(Metric):
    """Scale Invariant PSNR metric with local data range.

    Adapted from juglab/ScaleInvPSNR, this version of PSNR rescales the predictions and
    ground truth to have similar range, then computes the PSNR using each patch's data
    range.

    Parameters
    ----------
    n_channels : int
        Number of channels in the input images.
    **kwargs : Any
        Additional keyword arguments passed to the parent Metric class.

    Attributes
    ----------
    psnr_sum : Tensor
        Sum of PSNR values for each channel.
    total : Tensor
        Total number of samples processed.
    """

    def __init__(self, n_channels: int, **kwargs: Any):
        """Initialize the scale invariant PSNR metric.

        Parameters
        ----------
        n_channels : int
            Number of channels in the input images.
        **kwargs : Any
            Additional keyword arguments passed to the parent Metric class.
        """
        super().__init__(**kwargs)

        self.eps = torch.finfo(torch.float32).eps

        self.add_state(
            "psnr_sum",
            default=tensor([0.0 for _ in range(n_channels)]),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total",
            default=tensor([0.0 for _ in range(n_channels)]),
            dist_reduce_fx="sum",
        )

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update the metric states with values computed from a new batch.

        Parameters
        ----------
        preds : Tensor
            Predicted images tensor of shape (B, C, (Z), Y, X).
        target : Tensor
            Ground truth images tensor of shape (B, C, (Z), Y, X).
        """
        batch_size = target.shape[0]
        shape = target.shape
        dims = tuple(range(2, len(shape)))

        # compute min/max of the batches and channels
        batch_min = torch.amin(target, dim=dims)
        batch_max = torch.amax(target, dim=dims)
        data_range = batch_max - batch_min + self.eps

        # normalize range of gt and prediction
        tar_rescaled, pred_rescaled = normalise_range(target, preds)

        # compute mse
        mse = torch.mean((tar_rescaled - pred_rescaled) ** 2 + self.eps, dim=dims)

        # update states
        self.psnr_sum += torch.sum(10 * torch.log10(data_range**2 / mse), dim=0)
        self.total += batch_size

    def compute(self) -> Tensor | dict[str, Tensor]:
        """Compute the final metric value.

        Returns
        -------
        torch.Tensor
            Tensor of length C containing the computed PSNR for each channel.
        """
        return self.psnr_sum / self.total


class RangelessPSNR(Metric):
    """PSNR metric with local data range.

    PSNR often requires a user defined data range. This metric uses min and max of the
    batches to compute the data range. Equivalent to running
    `torchmetrics.image.PeakSignalNoiseRatio`, with `data_range` being the difference
    between each patch's max and min, on each patch individually.

    As opposed to `torchmetrics.image.PeakSignalNoiseRatio`, this implementation is
    compatible with 3D and multi-channel images.

    Parameters
    ----------
    n_channels : int
        Number of channels in the input images.
    **kwargs : Any
        Additional keyword arguments passed to the parent Metric class.

    Attributes
    ----------
    psnr_sum : Tensor
        Sum of PSNR values for each channel.
    total : Tensor
        Total number of samples processed.
    """

    def __init__(self, n_channels: int, **kwargs: Any):
        """Initialize the scale invariant PSNR metric.

        Parameters
        ----------
        n_channels : int
            Number of channels in the input images.
        **kwargs : Any
            Additional keyword arguments passed to the parent Metric class.
        """
        super().__init__(**kwargs)

        self.eps = torch.finfo(torch.float32).eps

        self.add_state(
            "psnr_sum",
            default=tensor([0.0 for _ in range(n_channels)]),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total",
            default=tensor([0.0 for _ in range(n_channels)]),
            dist_reduce_fx="sum",
        )

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update the metric states with values computed from a new batch.

        Parameters
        ----------
        preds : Tensor
            Predicted images tensor of shape (B, C, (Z), Y, X).
        target : Tensor
            Ground truth images tensor of shape (B, C, (Z), Y, X).
        """
        batch_size = target.shape[0]
        shape = target.shape
        dims = tuple(range(2, len(shape)))

        # compute min/max of the batches and channels
        batch_min = torch.amin(target, dim=dims)
        batch_max = torch.amax(target, dim=dims)
        data_range = batch_max - batch_min + self.eps

        # compute mse
        mse = torch.mean((target - preds) ** 2 + self.eps, dim=dims)

        # update states
        self.psnr_sum += torch.sum(10 * torch.log10(data_range**2 / mse), dim=0)
        self.total += batch_size

    def compute(self) -> Tensor | dict[str, Tensor]:
        """Compute the final metric value.

        Returns
        -------
        torch.Tensor
            Tensor of length C containing the computed PSNR for each channel.
        """
        return self.psnr_sum / self.total
