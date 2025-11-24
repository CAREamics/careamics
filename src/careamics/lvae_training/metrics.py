"""Metrics utilities for LVAE training and evaluation."""

from __future__ import annotations

import importlib
from collections import defaultdict
from collections.abc import Callable, Sequence

import numpy as np
import torch
from skimage.metrics import structural_similarity
from torch import Tensor
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

from careamics.models.lvae.utils import allow_numpy

try:
    _microssim = importlib.import_module("microssim")
except ImportError as exc:
    raise ImportError(
        "microssim is required for LVAE metrics; "
        "install it via `pip install microssim`."
    ) from exc

MicroMS3IM = _microssim.MicroMS3IM
MicroSSIM = _microssim.MicroSSIM

ArrayBatch = np.ndarray | Tensor
ArrayCollection = ArrayBatch | Sequence[np.ndarray]
ChannelStats = tuple[float, float]
HighSNRDict = dict[str, list[ChannelStats]]


class RunningPSNR:
    """Track the running PSNR over validation batches."""

    def __init__(self) -> None:
        self.mse_sum: Tensor
        self.count: float
        self.max_value: float | None
        self.min_value: float | None
        self.reset()

    def reset(self) -> None:
        """Reset accumulated statistics."""
        self.mse_sum = torch.tensor(0.0)
        self.count = 0.0
        self.max_value = None
        self.min_value = None

    def update(self, rec: Tensor, tar: Tensor) -> None:
        """Update statistics with a batch of reconstructed and target images."""
        ins_max = torch.max(tar).item()
        ins_min = torch.min(tar).item()
        if self.max_value is None:
            self.max_value = ins_max
            self.min_value = ins_min
        else:
            self.max_value = max(self.max_value, ins_max)
            self.min_value = min(self.min_value, ins_min)

        mse = (rec - tar) ** 2
        elementwise_mse = torch.mean(mse.view(len(mse), -1), dim=1)
        self.mse_sum = self.mse_sum + torch.nansum(elementwise_mse)
        invalid_elements = int(torch.sum(torch.isnan(elementwise_mse)).item())
        self.count += float(len(elementwise_mse) - invalid_elements)

    def get(self) -> Tensor | None:
        """Return the current PSNR."""
        if (
            self.count == 0
            or self.max_value is None
            or self.min_value is None
            or torch.isnan(self.mse_sum)
        ):
            return None
        rmse = torch.sqrt(self.mse_sum / self.count)
        return 20 * torch.log10((self.max_value - self.min_value) / rmse)


def zero_mean(x: Tensor) -> Tensor:
    """Return a zero-mean tensor along the channel dimension."""
    return x - torch.mean(x, dim=1, keepdim=True)


def fix_range(gt: Tensor, x: Tensor) -> Tensor:
    """Rescale a tensor to match the range of the ground truth."""
    denom = torch.sum(x * x, dim=1, keepdim=True)
    a = torch.sum(gt * x, dim=1, keepdim=True) / denom
    return x * a


def fix(gt: Tensor, x: Tensor) -> Tensor:
    """Zero-mean tensors and match prediction range to the ground truth."""
    gt_ = zero_mean(gt)
    return fix_range(gt_, zero_mean(x))


def compute_SE(arr: Sequence[float]) -> float:
    """Compute the standard error of the mean."""
    return float(np.std(arr) / np.sqrt(len(arr)))


def _psnr_internal(
    gt: Tensor,
    pred: Tensor,
    range_: Tensor | None = None,
) -> Tensor:
    """Compute PSNR given flattened tensors."""
    if range_ is None:
        range_ = torch.max(gt, dim=1).values - torch.min(gt, dim=1).values
    mse = torch.mean((gt - pred) ** 2, dim=1)
    return 20 * torch.log10(range_ / torch.sqrt(mse))


@allow_numpy
def PSNR(
    gt: Tensor,
    pred: Tensor,
    range_: Tensor | None = None,
) -> Tensor:
    """Compute PSNR for tensors shaped as (batch, H, W)."""
    if len(gt.shape) != 3:
        msg = "Images must be in shape: (batch, H, W)"
        raise ValueError(msg)
    gt_flat = gt.view(len(gt), -1)
    pred_flat = pred.view(len(gt), -1)
    return _psnr_internal(gt_flat, pred_flat, range_=range_)


@allow_numpy
def RangeInvariantPsnr(gt: Tensor, pred: Tensor) -> Tensor:
    """Compute range-invariant PSNR for grayscale images."""
    if len(gt.shape) != 3:
        msg = "Images must be in shape: (batch, H, W)"
        raise ValueError(msg)
    gt_flat = gt.view(len(gt), -1)
    pred_flat = pred.view(len(gt), -1)
    range_values = torch.max(gt_flat, dim=1).values - torch.min(gt_flat, dim=1).values
    ra = range_values / torch.std(gt_flat, dim=1)
    gt_norm = zero_mean(gt_flat) / torch.std(gt_flat, dim=1, keepdim=True)
    return _psnr_internal(zero_mean(gt_norm), fix(gt_norm, pred_flat), ra)


def _to_tensor_batch(data: ArrayCollection) -> Tensor:
    """Convert batch inputs to tensors."""
    if isinstance(data, Tensor):
        return data
    if isinstance(data, np.ndarray):
        return torch.as_tensor(data)
    return torch.as_tensor(np.stack(data, axis=0))


def _avg_psnr(
    target: ArrayCollection,
    prediction: ArrayCollection,
    psnr_fn: Callable[[Tensor, Tensor], Tensor],
) -> ChannelStats:
    """Return the mean PSNR and its standard error."""
    target_batch = _to_tensor_batch(target)
    prediction_batch = _to_tensor_batch(prediction)
    psnr_arr = [
        psnr_fn(
            target_batch[i][None] * 1.0,
            prediction_batch[i][None] * 1.0,
        ).item()
        for i in range(len(prediction_batch))
    ]
    mean_psnr = float(np.mean(psnr_arr))
    std_err_psnr = compute_SE(psnr_arr)
    return round(mean_psnr, 2), round(std_err_psnr, 3)


def avg_range_inv_psnr(
    target: ArrayCollection,
    prediction: ArrayCollection,
) -> ChannelStats:
    """Compute mean and standard error of range-invariant PSNR."""
    return _avg_psnr(target, prediction, RangeInvariantPsnr)


def avg_psnr(
    target: ArrayCollection,
    prediction: ArrayCollection,
) -> ChannelStats:
    """Compute mean and standard error of PSNR."""
    return _avg_psnr(target, prediction, PSNR)


def compute_masked_psnr(
    mask: np.ndarray,
    tar1: np.ndarray,
    tar2: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
) -> tuple[ChannelStats, ChannelStats]:
    """Compute PSNR on masked regions for two target/prediction pairs."""
    mask_bool = mask.astype(bool)[..., 0]
    tmp_tar1 = tar1[mask_bool].reshape((len(tar1), -1, 1))
    tmp_pred1 = pred1[mask_bool].reshape((len(tar1), -1, 1))
    tmp_tar2 = tar2[mask_bool].reshape((len(tar2), -1, 1))
    tmp_pred2 = pred2[mask_bool].reshape((len(tar2), -1, 1))
    psnr1 = avg_range_inv_psnr(tmp_tar1, tmp_pred1)
    psnr2 = avg_range_inv_psnr(tmp_tar2, tmp_pred2)
    return psnr1, psnr2


def avg_ssim(target: np.ndarray, prediction: np.ndarray) -> ChannelStats:
    """Compute mean and standard deviation of SSIM."""
    ssim_values = [
        structural_similarity(
            target[i],
            prediction[i],
            data_range=(target[i].max() - target[i].min()),
        )
        for i in range(len(target))
    ]
    return float(np.mean(ssim_values)), float(np.std(ssim_values))


@allow_numpy
def range_invariant_multiscale_ssim(
    gt_: ArrayBatch,
    pred_: ArrayBatch,
) -> float:
    """Compute range-invariant multiscale SSIM for one channel."""
    shape = gt_.shape
    gt_tensor = torch.as_tensor(gt_.reshape((shape[0], -1)))
    pred_tensor = torch.as_tensor(pred_.reshape((shape[0], -1)))
    gt_tensor = zero_mean(gt_tensor)
    pred_tensor = zero_mean(pred_tensor)
    pred_tensor = fix(gt_tensor, pred_tensor)
    pred_tensor = pred_tensor.reshape(shape)
    gt_tensor = gt_tensor.reshape(shape)
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(
        data_range=gt_tensor.max() - gt_tensor.min()
    )
    return ms_ssim(
        torch.as_tensor(pred_tensor[:, None]),
        torch.as_tensor(gt_tensor[:, None]),
    ).item()


def compute_multiscale_ssim(
    gt_: np.ndarray,
    pred_: np.ndarray,
    range_invariant: bool = True,
) -> list[ChannelStats]:
    """Compute channel-wise multiscale SSIM."""
    ms_ssim_values: dict[int, list[float]] = {i: [] for i in range(gt_.shape[-1])}
    for ch_idx in range(gt_.shape[-1]):
        tar_tmp = gt_[..., ch_idx]
        pred_tmp = pred_[..., ch_idx]
        if range_invariant:
            ms_ssim_values[ch_idx] = [
                range_invariant_multiscale_ssim(
                    tar_tmp[i : i + 1],
                    pred_tmp[i : i + 1],
                )
                for i in range(tar_tmp.shape[0])
            ]
        else:
            metric = MultiScaleStructuralSimilarityIndexMeasure(
                data_range=tar_tmp.max() - tar_tmp.min()
            )
            ms_ssim_values[ch_idx] = [
                metric(
                    torch.Tensor(pred_tmp[i : i + 1, None]),
                    torch.Tensor(tar_tmp[i : i + 1, None]),
                ).item()
                for i in range(tar_tmp.shape[0])
            ]

    return [
        (float(np.mean(ms_ssim_values[i])), compute_SE(ms_ssim_values[i]))
        for i in range(gt_.shape[-1])
    ]


def compute_custom_ssim(
    gt_: Sequence[np.ndarray],
    pred_: Sequence[np.ndarray],
    ssim_obj_dict: dict[int, MicroSSIM | MicroMS3IM],
) -> list[ChannelStats]:
    """Compute SSIM using custom per-channel scorers."""
    ms_ssim_values: dict[int, list[float]] = defaultdict(list)
    channels = gt_[0].shape[-1]
    for i in range(len(gt_)):
        for ch_idx in range(channels):
            tar_tmp = gt_[i][..., ch_idx]
            pred_tmp = pred_[i][..., ch_idx]
            ms_ssim_values[ch_idx].append(
                ssim_obj_dict[ch_idx].score(tar_tmp, pred_tmp)
            )
    return [
        (float(np.mean(ms_ssim_values[i])), compute_SE(ms_ssim_values[i]))
        for i in range(channels)
    ]


def _get_list_of_images_from_gt_pred(
    gt: np.ndarray | list[np.ndarray],
    pred: np.ndarray | list[np.ndarray],
    ch_idx: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return flattened lists of images for a given channel."""
    gt_list: list[np.ndarray] = []
    pred_list: list[np.ndarray] = []
    if isinstance(gt, list):
        for i in range(len(gt)):
            gt_tmp, pred_tmp = _get_list_of_images_from_gt_pred(gt[i], pred[i], ch_idx)
            gt_list += gt_tmp
            pred_list += pred_tmp
    elif isinstance(gt, np.ndarray):
        if len(gt.shape) == 3:
            return [gt[..., ch_idx] * 1.0], [pred[..., ch_idx]]
        if gt.shape != pred.shape:
            msg = f"gt shape: {gt.shape}, pred shape: {pred.shape}"
            raise ValueError(msg)
        for n_idx in range(gt.shape[0]):
            gt_tmp, pred_tmp = _get_list_of_images_from_gt_pred(
                gt[n_idx],
                pred[n_idx],
                ch_idx,
            )
            gt_list += gt_tmp
            pred_list += pred_tmp
    return gt_list, pred_list


def compute_stats(
    highres_data: Sequence[np.ndarray],
    pred_unnorm: Sequence[np.ndarray],
    verbose: bool = True,
) -> HighSNRDict:
    """Compute PSNR- and SSIM-based metrics on high-SNR data."""
    psnr_list: list[ChannelStats] = []
    microssim_list: list[ChannelStats] = []
    ms3im_list: list[ChannelStats] = []
    ssim_list: list[ChannelStats] = []
    msssim_list: list[ChannelStats] = []

    channel_count = highres_data[0].shape[-1]
    for ch_idx in range(channel_count):
        gt_ch, pred_ch = _get_list_of_images_from_gt_pred(
            list(highres_data),
            list(pred_unnorm),
            ch_idx,
        )
        psnr_list.append(avg_range_inv_psnr(gt_ch, pred_ch))

        microssim_obj = MicroSSIM()
        microssim_obj.fit(gt_ch, pred_ch)
        mssim_scores = [
            microssim_obj.score(gt_ch[i], pred_ch[i]) for i in range(len(gt_ch))
        ]
        microssim_list.append(
            (float(np.mean(mssim_scores)), compute_SE(mssim_scores))
        )

        m3sim_obj = MicroMS3IM()
        m3sim_obj.fit(gt_ch, pred_ch)
        ms3im_scores = [
            m3sim_obj.score(gt_ch[i], pred_ch[i]) for i in range(len(gt_ch))
        ]
        ms3im_list.append((float(np.mean(ms3im_scores)), compute_SE(ms3im_scores)))

        ssim_scores = [
            structural_similarity(
                gt_ch[i],
                pred_ch[i],
                data_range=gt_ch[i].max() - gt_ch[i].min(),
            )
            for i in range(len(gt_ch))
        ]
        ssim_list.append((float(np.mean(ssim_scores)), compute_SE(ssim_scores)))

        ms_ssim_scores = []
        for i in range(len(gt_ch)):
            metric = MultiScaleStructuralSimilarityIndexMeasure(
                data_range=gt_ch[i].max() - gt_ch[i].min()
            )
            ms_ssim_scores.append(
                metric(
                    torch.Tensor(pred_ch[i][None, None]),
                    torch.Tensor(gt_ch[i][None, None]),
                ).item()
            )
        msssim_list.append((float(np.mean(ms_ssim_scores)), compute_SE(ms_ssim_scores)))

    if verbose:
        def ssim_str(values: ChannelStats) -> str:
            return f"{np.round(values[0], 3):.3f}+-{np.round(values[1], 3):.3f}"

        def psnr_str(values: ChannelStats) -> str:
            return f"{np.round(values[0], 2)}+-{np.round(values[1], 3)}"

        print(
            "PSNR on Highres",
            "\t".join(psnr_str(value) for value in psnr_list),
        )
        print(
            "MicroSSIM on Highres",
            "\t".join(ssim_str(value) for value in microssim_list),
        )
        print(
            "MicroS3IM on Highres",
            "\t".join(ssim_str(value) for value in ms3im_list),
        )
        print(
            "SSIM on Highres",
            "\t".join(ssim_str(value) for value in ssim_list),
        )
        print(
            "MSSSIM on Highres",
            "\t".join(ssim_str(value) for value in msssim_list),
        )

    return {
        "rangeinvpsnr": psnr_list,
        "microssim": microssim_list,
        "ms3im": ms3im_list,
        "ssim": ssim_list,
        "msssim": msssim_list,
    }
