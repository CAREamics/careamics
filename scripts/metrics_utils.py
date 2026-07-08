"""Global image-quality metrics for stitched MicroSplit predictions.

Ported (and trimmed) from ``lambdaSplit/utils/metrics_utils.py`` for use by the
tiling-experiment scripts. Provides channel-wise metric computers, an
orchestrator (:func:`compute_unmixing_metrics`), and JSON log / load / display
helpers.

The metric primitives (``scale_invariant_psnr``, ``range_invariant_multiscale_ssim``,
``lpips``) live in :mod:`careamics.metrics.metrics`; ``MicroMS3IM`` comes from
``microssim`` and Pearson correlation from ``skimage``.

Output structure (consumed by ``scripts.plotting_utils._parse_metrics``):
- ``metrics_avg``: ``{metric_group: {channel_key: value}}`` where ``channel_key``
  is e.g. ``"RI-PSNR_FP#1"`` or ``"Avg RI-PSNR"``.
- ``metrics_per_img``: same, with an extra innermost ``{img_fname: value}`` level.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Union
from warnings import warn

import numpy as np
from microssim import MicroMS3IM
from numpy.typing import NDArray
from skimage.measure import pearson_corr_coeff
from tqdm import tqdm

from careamics.metrics.metrics import (
    lpips,
    range_invariant_multiscale_ssim,
    scale_invariant_psnr,
)


def _normalize_for_lpips(imgs: list[NDArray]) -> list[NDArray]:
    """Normalize the given images in [0, 1] for LPIPS.

    Parameters
    ----------
    imgs : list[NDArray]
        A list of multi-channel images to normalize, each of shape (C, [Z], Y, X).

    Returns
    -------
    list[NDArray]
        The normalized images.
    """
    # TODO: use training dset stats for normalization (?)
    ax_idxs = tuple(range(1, imgs[0].ndim))
    min_ = np.min([img.min(axis=ax_idxs) for img in imgs])
    max_ = np.max([img.max(axis=ax_idxs) for img in imgs])
    min_ = np.asarray(min_).reshape(-1, *np.ones_like(ax_idxs, dtype=int))
    max_ = np.asarray(max_).reshape(-1, *np.ones_like(ax_idxs, dtype=int))
    return [(img - min_) / (max_ - min_) for img in imgs]


def _compute_channelwise_psnr(
    pred_imgs: list[NDArray],
    gt_imgs: list[NDArray],
    img_fnames: list[str] = None,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Compute the channel-wise PSNR over the given images.

    Parameters
    ----------
    pred_imgs : list[NDArray]
        The predicted data as a list of arrays, each of shape (C, [Z], Y, X).
    gt_imgs : list[NDArray]
        The ground truth data as a list of arrays, each of shape (C, [Z], Y, X).
    img_fnames : list[str]
        The names of the images, used for per-image values. Default is None.

    Returns
    -------
    tuple[dict[str, float], dict[str, dict[str, float]]]
        The PSNR values for each channel plus the average one, both as average
        metrics over the dataset and per-image metrics.
    """
    psnr_avg_dict = {}
    psnr_per_img_dict = {}
    for ch in tqdm(range(pred_imgs[0].shape[0]), desc="Computing channel-wise PSNR"):

        ch_psnr_lst = []
        for i in range(len(pred_imgs)):  # iterate over images
            ch_psnr_lst.append(
                scale_invariant_psnr(gt=gt_imgs[i][ch], pred=pred_imgs[i][ch])
            )

        psnr_avg_dict[f"RI-PSNR_FP#{ch+1}"] = np.mean(ch_psnr_lst)
        psnr_per_img_dict[f"RI-PSNR_FP#{ch+1}"] = {
            k: v for k, v in zip(img_fnames, ch_psnr_lst)
        }

    # --- get average over all channels
    # avg metric
    psnr_avg_dict["Avg RI-PSNR"] = np.mean([v for v in psnr_avg_dict.values()])
    # per-image metric
    values = (
        np.array(
            [[v for v in ch_dict.values()] for ch_dict in psnr_per_img_dict.values()]
        )
        .mean(axis=0)
        .tolist()
    )
    psnr_per_img_dict["Avg RI-PSNR"] = {k: v for k, v in zip(img_fnames, values)}

    return psnr_avg_dict, psnr_per_img_dict


def _compute_channelwise_lpips(
    pred_imgs: list[NDArray],
    gt_imgs: list[NDArray],
    img_fnames: list[str] = None,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Compute the channel-wise LPIPS over the given images.

    Parameters
    ----------
    pred_imgs : list[NDArray]
        The predicted data as a list of arrays. Each array has shape (C, [Z], Y, X).
    gt_imgs : list[NDArray]
        The ground truth data as a list of arrays. Each array has shape (C, [Z], Y, X).
    img_fnames : list[str]
        The names of the images, used for per-image values. Default is None.

    Returns
    -------
    tuple[dict[str, float], dict[str, dict[str, float]]]
        The LPIPS values for each channel plus the average one, both as average
        metrics over the dataset and per-image metrics.
    """
    # normalize the images in [0, 1]
    pred_imgs = _normalize_for_lpips(pred_imgs)
    gt_imgs = _normalize_for_lpips(gt_imgs)

    lpips_avg_dict = {}
    lpips_per_img_dict = {}
    for ch in tqdm(range(pred_imgs[0].shape[0]), desc="Computing channel-wise LPIPS"):

        ch_lpips_lst = []
        for i in range(len(pred_imgs)):  # iterate over images
            # inputs are expected to be RGB + have batch dimension
            curr_target = np.repeat(gt_imgs[i][ch : ch + 1], repeats=3, axis=0)[
                None, ...
            ]
            curr_pred = np.repeat(pred_imgs[i][ch : ch + 1], repeats=3, axis=0)[
                None, ...
            ]
            ch_lpips_lst.append(lpips(prediction=curr_pred, target=curr_target))

        lpips_avg_dict[f"LPIPS_FP#{ch+1}"] = np.mean(ch_lpips_lst)
        lpips_per_img_dict[f"LPIPS_FP#{ch+1}"] = {
            k: v for k, v in zip(img_fnames, ch_lpips_lst)
        }

    # --- get average over all channels
    # avg metric
    lpips_avg_dict["Avg LPIPS"] = np.mean([v for v in lpips_avg_dict.values()])
    # per-image metric
    values = (
        np.array(
            [[v for v in ch_dict.values()] for ch_dict in lpips_per_img_dict.values()]
        )
        .mean(axis=0)
        .tolist()
    )
    lpips_per_img_dict["Avg LPIPS"] = {k: v for k, v in zip(img_fnames, values)}

    return lpips_avg_dict, lpips_per_img_dict


def _compute_channelwise_multiscale_ssim(
    pred_imgs: list[NDArray],
    gt_imgs: list[NDArray],
    img_fnames: list[str] = None,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Compute the channel-wise multiscale SSIM over the given images.

    Parameters
    ----------
    pred_imgs : list[NDArray]
        The predicted data as a list of arrays, each of shape (C, [Z], Y, X).
    gt_imgs : list[NDArray]
        The ground truth data as a list of arrays, each of shape (C, [Z], Y, X).
    img_fnames : list[str]
        The names of the images, used for per-image values. Default is None.

    Returns
    -------
    tuple[dict[str, float], dict[str, dict[str, float]]]
        The MS-SSIM values for each channel plus the average one, both as average
        metrics over the dataset and per-image metrics.
    """
    ssim_avg_dict = {}
    ssim_per_img_dict = {}
    for ch in tqdm(range(pred_imgs[0].shape[0]), desc="Computing channel-wise MSSIM"):

        ch_ssim_lst = []
        for i in range(len(pred_imgs)):  # iterate over images
            ch_ssim_lst.append(
                range_invariant_multiscale_ssim(
                    gt_=gt_imgs[i][ch][None, ...],  # requires batch dimension
                    pred_=pred_imgs[i][ch][None, ...],
                )
            )

        ssim_avg_dict[f"RI-MSSIM_FP#{ch+1}"] = np.mean(ch_ssim_lst)
        ssim_per_img_dict[f"RI-MSSIM_FP#{ch+1}"] = {
            k: v for k, v in zip(img_fnames, ch_ssim_lst)
        }

    # --- get average over all channels
    # avg metric
    ssim_avg_dict["Avg RI-MSSIM"] = np.mean([v for v in ssim_avg_dict.values()])
    # per-image metric
    values = (
        np.array(
            [[v for v in ch_dict.values()] for ch_dict in ssim_per_img_dict.values()]
        )
        .mean(axis=0)
        .tolist()
    )
    ssim_per_img_dict["Avg RI-MSSIM"] = {k: v for k, v in zip(img_fnames, values)}

    return ssim_avg_dict, ssim_per_img_dict


def _compute_channelwise_microms3im(
    pred_imgs: list[NDArray],
    gt_imgs: list[NDArray],
    img_fnames: list[str] = None,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Compute the channel-wise MicroMS3IM metric over the given images.

    For each channel, a MicroMS3IM scaler is fitted over all images for that
    channel, ensuring consistent scaling within each channel. The fitted scaler
    is then used to score each image individually.

    Parameters
    ----------
    pred_imgs : list[NDArray]
        The predicted data as a list of arrays. Each array has shape (C, [Z], Y, X).
    gt_imgs : list[NDArray]
        The ground truth data as a list of arrays. Each array has shape (C, [Z], Y, X).
    img_fnames : list[str]
        The names of the images, used for per-image values. Default is None.

    Returns
    -------
    tuple[dict[str, float], dict[str, dict[str, float]]]
        The MicroMS3IM values for each channel plus the average one, both as
        average metrics over the dataset and per-image metrics.
    """
    if pred_imgs[0].ndim == 4:
        warn(
            "MicroMS3IM doesn't work for 3D data. Skipping computation of this metric."
        )
        return None, None

    microms3im_avg_dict = {}
    microms3im_per_img_dict = {}
    for ch in tqdm(
        range(pred_imgs[0].shape[0]), desc="Computing channel-wise MicroMS3IM"
    ):

        # Stack all images for this channel: shape (N_images, [Z], Y, X)
        gt_ch_lst = [img[ch] for img in gt_imgs]
        pred_ch_lst = [img[ch] for img in pred_imgs]

        # Fit MicroMS3IM scaler on all images for this channel
        microssim = MicroMS3IM()
        microssim.fit(gt_ch_lst, pred_ch_lst)

        # Score each image individually using the fitted scaler
        ch_microms3im_lst = []
        for i in range(len(pred_imgs)):
            score = microssim.score(gt_imgs[i][ch], pred_imgs[i][ch])
            ch_microms3im_lst.append(score.detach().cpu().item())

        microms3im_avg_dict[f"MicroMS3IM_FP#{ch+1}"] = np.mean(ch_microms3im_lst)
        microms3im_per_img_dict[f"MicroMS3IM_FP#{ch+1}"] = {
            k: v for k, v in zip(img_fnames, ch_microms3im_lst)
        }

    # --- get average over all channels
    # avg metric
    microms3im_avg_dict["Avg MicroMS3IM"] = np.mean(
        [v for v in microms3im_avg_dict.values()]
    )
    # per-image metric
    values = (
        np.array(
            [
                [v for v in ch_dict.values()]
                for ch_dict in microms3im_per_img_dict.values()
            ]
        )
        .mean(axis=0)
        .tolist()
    )
    microms3im_per_img_dict["Avg MicroMS3IM"] = {
        k: v for k, v in zip(img_fnames, values)
    }

    return microms3im_avg_dict, microms3im_per_img_dict


def _compute_channelwise_pearson(
    pred_imgs: list[NDArray],
    gt_imgs: list[NDArray],
    img_fnames: list[str] = None,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Compute the channel-wise Pearson correlation over the given images.

    Parameters
    ----------
    pred_imgs : list[NDArray]
        The predicted data as a list of arrays, each of shape (C, [Z], Y, X).
    gt_imgs : list[NDArray]
        The ground truth data as a list of arrays, each of shape (C, [Z], Y, X).
    img_fnames : list[str]
        The names of the images, used for per-image values. Default is None.

    Returns
    -------
    tuple[dict[str, float], dict[str, dict[str, float]]]
        The Pearson correlation values for each channel plus the average one, both
        as average metrics over the dataset and per-image metrics.
    """
    pearson_avg_dict = {}
    pearson_per_img_dict = {}

    for ch in tqdm(range(pred_imgs[0].shape[0]), desc="Computing channel-wise Pearson"):
        ch_pearson_lst = []
        for i in range(len(pred_imgs)):  # iterate over images
            ch_pearson_lst.append(
                pearson_corr_coeff(pred_imgs[i][ch], gt_imgs[i][ch])[0]
            )

        pearson_avg_dict[f"Pearson_FP#{ch+1}"] = np.mean(ch_pearson_lst)
        pearson_per_img_dict[f"Pearson_FP#{ch+1}"] = {
            k: v for k, v in zip(img_fnames, ch_pearson_lst)
        }

    # --- get average over all channels
    # avg metric
    pearson_avg_dict["Avg Pearson"] = np.mean([v for v in pearson_avg_dict.values()])
    # per-image metric
    values = (
        np.array(
            [[v for v in ch_dict.values()] for ch_dict in pearson_per_img_dict.values()]
        )
        .mean(axis=0)
        .tolist()
    )
    pearson_per_img_dict["Avg Pearson"] = {k: v for k, v in zip(img_fnames, values)}

    return pearson_avg_dict, pearson_per_img_dict


def compute_unmixing_metrics(
    pred_imgs: list[NDArray],
    gt_imgs: list[NDArray],
    metrics: list[Literal["PSNR", "LPIPS", "MSSIM", "MicroMS3IM", "Pearson"]],
    img_fnames: list[str],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, dict[str, float]]]]:
    """Compute metrics for the current unmixing experiment.

    Parameters
    ----------
    pred_imgs : list[NDArray]
        The full-frame unmixed predictions, each of shape (C, [Z], Y, X).
    gt_imgs : list[NDArray]
        The ground truth unmixed data, each of shape (C, [Z], Y, X).
    metrics : list[Literal["PSNR", "LPIPS", "MSSIM", "MicroMS3IM", "Pearson"]]
        The list of metrics to compute.
    img_fnames : list[str]
        The names of the images, used for per-image values.

    Returns
    -------
    tuple[dict[str, dict[str, float]], dict[str, dict[str, dict[str, float]]]]
        The computed metrics in two dictionaries, the first one with average
        metrics over the dataset, and the second with per-image metrics. Outer
        dictionary keys are the metric names. For dataset metrics, inner dictionary
        keys are channel ids or "Avg", with the corresponding metric values. For
        per-image values, the inner dictionaries contain another level of nested
        dictionaries with image names as keys and metric values as values.
    """
    assert len(pred_imgs) == len(
        gt_imgs
    ), "Number of predicted and GT images must match."
    assert len(pred_imgs[0].shape) == len(
        gt_imgs[0].shape
    ), "Predictions and GT images must have the same dimensions."

    # exclude empty images from metrics computation
    n_spatial_dim = pred_imgs[0].ndim - 1
    non_empty_idxs = [
        i
        for i in range(len(gt_imgs))
        if not np.any(np.isclose(gt_imgs[i].sum(tuple(range(1, n_spatial_dim + 1))), 0))
    ]
    pred_imgs = [pred_imgs[i] for i in non_empty_idxs]
    gt_imgs = [gt_imgs[i] for i in non_empty_idxs]
    img_fnames = [img_fnames[i] for i in non_empty_idxs]

    metrics_avg_dict = {}
    metrics_per_img_dict = {}
    # --- PSNR
    # TODO: this can be optimized, but it's fine for now
    if "PSNR" in metrics:
        psnr_avg_dict, psnr_per_img_dict = _compute_channelwise_psnr(
            pred_imgs, gt_imgs, img_fnames
        )
        metrics_avg_dict["PSNR"] = psnr_avg_dict
        metrics_per_img_dict["PSNR"] = psnr_per_img_dict

    # --- LPIPS
    if "LPIPS" in metrics:
        lpips_avg_dict, lpips_per_img_dict = _compute_channelwise_lpips(
            pred_imgs, gt_imgs, img_fnames
        )
        metrics_avg_dict["LPIPS"] = lpips_avg_dict
        metrics_per_img_dict["LPIPS"] = lpips_per_img_dict

    # --- Multiscale SSIM (MSSIM)
    if "MSSIM" in metrics:
        ssim_avg_dict, ssim_per_img = _compute_channelwise_multiscale_ssim(
            pred_imgs, gt_imgs, img_fnames
        )
        metrics_avg_dict["MSSIM"] = ssim_avg_dict
        metrics_per_img_dict["MSSIM"] = ssim_per_img

    # --- MicroMS3IM
    if "MicroMS3IM" in metrics:
        microms3im_avg_dict, microms3im_per_img = _compute_channelwise_microms3im(
            pred_imgs, gt_imgs, img_fnames
        )
        if microms3im_avg_dict is not None:
            metrics_avg_dict["MicroMS3IM"] = microms3im_avg_dict
            metrics_per_img_dict["MicroMS3IM"] = microms3im_per_img

    # --- Pearson correlation
    if "Pearson" in metrics:
        pearson_avg_dict, pearson_per_img_dict = _compute_channelwise_pearson(
            pred_imgs, gt_imgs, img_fnames
        )
        metrics_avg_dict["Pearson"] = pearson_avg_dict
        metrics_per_img_dict["Pearson"] = pearson_per_img_dict

    show_metrics(metrics_avg_dict)
    return metrics_avg_dict, metrics_per_img_dict


def log_metrics(
    metrics_dict: dict[str, dict[str, float]],
    log_dir: Union[str, Path],
    data_path: Union[str, Path],
    *,
    mmse_count: Optional[int] = None,
    ckpt_name: Optional[str] = None,
    tile_size: Optional[Union[list[int], str]] = None,
    tile_overlap: Optional[Union[list[int], str]] = None,
    per_image: bool = False,
    filename: Optional[str] = None,
) -> None:
    """Log the metrics in a JSON file.

    Parameters
    ----------
    metrics_dict : dict[str, dict[str, float]]
        Dictionary containing the metrics.
    log_dir : Union[str, Path]
        Directory where the metrics are saved.
    data_path : Union[str, Path]
        Path to the evaluated data, stored for provenance.
    mmse_count : Optional[int]
        Number of samples used for MMSE estimation. Stored for provenance.
    ckpt_name : Optional[str]
        Checkpoint name. Stored for provenance.
    tile_size : Optional[Union[list[int], str]]
        Tile size used at inference. Stored for provenance.
    tile_overlap : Optional[Union[list[int], str]]
        Tile overlap (or stride) used at inference. Stored for provenance.
    per_image : bool
        Whether the metrics are per image or not. Used for the default filename.
    filename : Optional[str]
        Custom filename for the metrics file. If None, defaults to
        "metrics_per_image.json" or "metrics.json" based on `per_image`.
    """
    if filename is not None:
        metrics_fpath = os.path.join(log_dir, filename)
    elif per_image:
        metrics_fpath = os.path.join(log_dir, "metrics_per_image.json")
    else:
        metrics_fpath = os.path.join(log_dir, "metrics.json")
    print(f"\nLogging metrics to {metrics_fpath}")

    timestamp = datetime.now().strftime("%d/%m/%y_%H:%M")
    logs_dict = {
        "timestamp": timestamp,
        "MMSE_count": mmse_count,
        "ckpt_name": ckpt_name,
        "tile_size": str(tile_size),
        "tile_overlap": str(tile_overlap),
        "data_path": str(data_path),
        "metrics": metrics_dict,
    }
    if os.path.exists(metrics_fpath):
        with open(metrics_fpath, "r") as f:
            metrics = json.load(f)
        metrics["evaluations"].append(logs_dict)
    else:
        metrics = {"evaluations": [logs_dict]}
    with open(metrics_fpath, "w") as f:
        json.dump(metrics, f, indent=4)


def load_metrics(
    log_dir: Union[str, Path],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, dict[str, float]]]]:
    """Load the pre-computed metrics from disk.

    Parameters
    ----------
    log_dir : Union[str, Path]
        Directory where the logs are saved.

    Returns
    -------
    tuple[dict[str, dict[str, float]], dict[str, dict[str, dict[str, float]]]]
        The loaded metrics dicts, the first one with average metrics over the
        dataset, and the second with per-image metrics.
    """
    metrics_per_img_fpath = os.path.join(log_dir, "metrics_per_image.json")
    metrics_avg_fpath = os.path.join(log_dir, "metrics.json")

    print(f"Loading pre-computed average metrics at {metrics_avg_fpath}\n")
    print(f"Loading pre-computed per-image metrics at {metrics_per_img_fpath}\n")

    with open(metrics_avg_fpath, "r") as f:
        metrics = json.load(f)
        metric_avg = metrics["evaluations"][-1]["metrics"]

    with open(metrics_per_img_fpath, "r") as f:
        metrics = json.load(f)
        metric_per_img = metrics["evaluations"][-1]["metrics"]

    return metric_avg, metric_per_img


def show_metrics(metrics_dict: dict[str, dict[str, float]]) -> None:
    """Print the metrics in a nice format.

    Parameters
    ----------
    metrics_dict : dict[str, dict[str, float]]
        Dictionary containing the metrics.
    """
    print("\nModel's outputs")
    print("---------------")
    for metric, subdict in metrics_dict.items():
        print(f"{metric}")
        for key, value in subdict.items():
            print(f"-> {key}: {value:.3f}")
        print("---------------\n")
