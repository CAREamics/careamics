"""MicroSplit — compute global metrics on stitched predictions (CL-args entry point).

Companion to the inference scripts (:mod:`scripts.microsplit_inner_tiling_inference`
and :mod:`scripts.microsplit_switi_inference`). Those scripts save, per prediction
directory, a ``predictions.npz`` (keyed by input-image filename stem, each stitched
prediction shaped ``(S, C, [Z], Y, X)`` and already denormalised to target units)
alongside an ``inference_params.json`` recording the run configuration:

    {"tile_size": [...], "overlap": [...], "mmse_count": N,
     "ckpt_path": "...", "data_dir": "..."}

This script reads that JSON to locate the GT (``<data_dir>/targets/<split>/``) and
to fill the provenance fields of the metric logs, loads the matching target TIFFs,
and computes channel-wise global metrics (PSNR, LPIPS, MS-SSIM, MicroMS3IM,
Pearson) via :func:`scripts.metrics_utils.compute_unmixing_metrics`. Results are
written next to the predictions as ``metrics.json`` (dataset averages) and
``metrics_per_image.json``. Whether the data is 3D is inferred from
``len(tile_size)`` (3 → 3D, 2 → 2D), so no pkl config is loaded.

Prediction ↔ GT matching is positional: ``inputs/<split>`` and ``targets/<split>``
are both sorted (see :func:`scripts.io.list_files`), and predictions are keyed by
the input stem, so prediction ``i`` pairs with target file ``i``. When a file holds
several frames (``S > 1``) each frame is scored as a separate image.

Run from the repo root (``--pred-dir`` holds ``predictions.npz`` +
``inference_params.json``):

    python -m scripts.microsplit_compute_metrics \\
        --pred-dir /project/careamics/switi/results/HT_LIF24_5ms/predictions_MMSE64/sw_inner_tiling
    python -m scripts.microsplit_compute_metrics \\
        --pred-dir /project/careamics/switi/results/CARE3D_liver/predictions_MMSE64/inner_tiling \\
        --metrics PSNR MSSIM Pearson
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scripts.io import list_files
from scripts.metrics_utils import compute_unmixing_metrics, log_metrics
from scripts.stats import _load_canonical


def _ensure_canonical(arr: NDArray, *, is_3d: bool) -> NDArray:
    """Pad leading axes so ``arr`` has shape ``(S, C, [Z], Y, X)``.

    Stitched predictions are already full ``SC(Z)YX``; this only prepends size-1
    axes when a prediction was stored without an explicit sample axis.
    """
    spatial = 3 if is_3d else 2
    target_ndim = spatial + 2  # S, C, plus spatial
    extra = target_ndim - arr.ndim
    if extra < 0 or extra > 2:
        raise ValueError(
            f"Unexpected prediction ndim {arr.ndim} for "
            f"{'3D' if is_3d else '2D'} data: expected one of "
            f"{{{spatial}, {spatial + 1}, {spatial + 2}}}."
        )
    for _ in range(extra):
        arr = arr[np.newaxis]
    return arr


def _center_crop_to_match(pred: NDArray, gt: NDArray) -> tuple[NDArray, NDArray]:
    """Center-crop ``pred`` and ``gt`` (each ``(C, [Z], Y, X)``) to a common shape.

    Spatial dims can differ by a few pixels if the stitched prediction retains
    tiling padding while the GT is at its original size. Cropping both to the
    per-axis minimum around the center keeps them aligned. The channel axis
    (index 0) is left untouched.
    """
    if pred.shape == gt.shape:
        return pred, gt
    pred_slices: list[slice] = [slice(None)]
    gt_slices: list[slice] = [slice(None)]
    for dp, dg in zip(pred.shape[1:], gt.shape[1:], strict=True):
        d = min(dp, dg)
        pred_slices.append(slice((dp - d) // 2, (dp - d) // 2 + d))
        gt_slices.append(slice((dg - d) // 2, (dg - d) // 2 + d))
    return pred[tuple(pred_slices)], gt[tuple(gt_slices)]


def _load_pred_gt_pairs(
    predictions_path: Path,
    data_dir: Path,
    split: str,
    *,
    is_3d: bool,
) -> tuple[list[NDArray], list[NDArray], list[str]]:
    """Load predictions and matching GT targets as parallel per-image lists.

    Returns ``(pred_imgs, gt_imgs, img_fnames)`` where each image is
    ``(C, [Z], Y, X)``; multi-frame files (``S > 1``) are unrolled into one image
    per frame with a ``<stem>__s{n}`` name.
    """
    npz = np.load(predictions_path)
    npz_keys = set(npz.keys())

    input_files = list_files(data_dir, split, "inputs")
    target_files = list_files(data_dir, split, "targets")
    if len(input_files) != len(target_files):
        raise ValueError(
            f"Input/target count mismatch for split {split!r}: "
            f"{len(input_files)} inputs vs {len(target_files)} targets."
        )

    pred_imgs: list[NDArray] = []
    gt_imgs: list[NDArray] = []
    img_fnames: list[str] = []
    for i, (inp, tgt) in enumerate(zip(input_files, target_files, strict=True)):
        # Predictions are keyed by input stem; fall back to the array-source key.
        key = inp.stem if inp.stem in npz_keys else f"pred_{i:04d}"
        if key not in npz_keys:
            raise KeyError(
                f"No prediction for input {inp.name!r} (tried keys "
                f"{inp.stem!r} and pred_{i:04d}) in {predictions_path}."
            )
        pred = _ensure_canonical(npz[key], is_3d=is_3d).astype(np.float32)
        gt = _load_canonical(tgt, is_3d=is_3d).astype(np.float32)

        if pred.shape[1] != gt.shape[1]:
            raise ValueError(
                f"Channel mismatch for {inp.stem!r}: prediction has "
                f"{pred.shape[1]} channels, target has {gt.shape[1]}."
            )
        if pred.shape[0] != gt.shape[0]:
            raise ValueError(
                f"Sample-count (S) mismatch for {inp.stem!r}: prediction S="
                f"{pred.shape[0]}, target S={gt.shape[0]}."
            )

        n_samples = pred.shape[0]
        for s in range(n_samples):
            p_img, g_img = _center_crop_to_match(pred[s], gt[s])
            pred_imgs.append(p_img)
            gt_imgs.append(g_img)
            img_fnames.append(inp.stem if n_samples == 1 else f"{inp.stem}__s{s}")

    npz.close()
    return pred_imgs, gt_imgs, img_fnames


def main(args: argparse.Namespace) -> Path:
    """Compute and log metrics for one prediction directory.

    Returns the directory where the metric JSONs were written.
    """
    pred_dir = args.pred_dir
    predictions_path = pred_dir / args.predictions_filename
    params_path = pred_dir / args.params_filename
    if not predictions_path.is_file():
        raise SystemExit(f"Predictions file not found: {predictions_path}")
    if not params_path.is_file():
        raise SystemExit(f"Inference-params file not found: {params_path}")

    params = json.loads(params_path.read_text())
    tile_size = params["tile_size"]
    is_3d = len(tile_size) == 3  # [Z, Y, X] -> 3D, [Y, X] -> 2D
    data_dir = Path(params["data_dir"])

    print(f"Loading predictions from {predictions_path}")
    print(f"Loading GT targets from  {data_dir / 'targets' / args.split}")
    pred_imgs, gt_imgs, img_fnames = _load_pred_gt_pairs(
        predictions_path, data_dir, args.split, is_3d=is_3d
    )
    print(f"matched {len(pred_imgs)} image(s), shape per image {pred_imgs[0].shape}")

    metrics_avg, metrics_per_img = compute_unmixing_metrics(
        pred_imgs=pred_imgs,
        gt_imgs=gt_imgs,
        metrics=args.metrics,
        img_fnames=img_fnames,
    )

    # Log both dataset-average and per-image metrics next to the predictions,
    # filling provenance fields straight from inference_params.json.
    ckpt_path = params.get("ckpt_path")
    common = dict(
        log_dir=pred_dir,
        data_path=data_dir / "targets" / args.split,
        mmse_count=params.get("mmse_count"),
        ckpt_name=Path(ckpt_path).name if ckpt_path else None,
        tile_size=tile_size,
        tile_overlap=params.get("overlap"),
    )
    log_metrics(metrics_avg, per_image=False, **common)
    log_metrics(metrics_per_img, per_image=True, **common)
    return pred_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="microsplit-compute-metrics",
        description=(
            "Compute channel-wise global metrics (PSNR, LPIPS, MS-SSIM, "
            "MicroMS3IM, Pearson) on stitched MicroSplit predictions vs. GT "
            "targets, writing metrics.json / metrics_per_image.json next to the "
            "predictions. Run configuration is read from inference_params.json."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--pred-dir",
        required=True,
        type=Path,
        help="prediction directory holding predictions.npz and "
        "inference_params.json (e.g. "
        "<results>/<dataset>/predictions_MMSE64/inner_tiling)",
    )
    p.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="which on-disk split was predicted",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        default=["PSNR", "LPIPS", "MSSIM", "MicroMS3IM", "Pearson"],
        choices=["PSNR", "LPIPS", "MSSIM", "MicroMS3IM", "Pearson"],
        help="metrics to compute (MicroMS3IM is auto-skipped for 3D data)",
    )
    p.add_argument(
        "--predictions-filename",
        default="predictions.npz",
        help="NPZ filename inside the prediction directory",
    )
    p.add_argument(
        "--params-filename",
        default="inference_params.json",
        help="inference-params JSON filename inside the prediction directory",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
