"""MicroSplit — sliding-window tiled inference (CL-args entry point).

Headless companion to :mod:`scripts.microsplit_sw_tiled_inference`. Loads a
pre-trained MicroSplit checkpoint, runs prediction with the dense
`SlidingWindowTiledPatching` strategy (overlapping kept regions, uniform
per-pixel MMSE coverage `K^d`), stitches by averaging via the streaming
`sw_tiled_prediction` helper, and saves a single `.npz` keyed by input-image
identifier.

Stride is **derived** from `--mmse-count` (target effective per-pixel coverage)
together with the patch size (read from the legacy `config.pkl`) and
`--overlap`. The YX stride is constrained symmetric; for 3D the caller fixes
`--stride-z` explicitly and the YX subproblem solves
`K_y * K_x >= ceil(MMSE_COUNT / K_z)`.

The model's per-tile draws are hardcoded to 1 (canonical for SW: per-pixel
coverage comes from the patching geometry).

Run from the repo root:

    python -m scripts.microsplit_sw_tiled_predict --dataset HT_LIF24_5ms \\
        --mmse-count 64
    python -m scripts.microsplit_sw_tiled_predict --dataset CARE3D_liver \\
        --overlap 0 32 32 --mmse-count 320 --stride-z 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from careamics.dev.sliding_window_tiled_pred import (
    compute_stride_for_mmse_count,
    sw_tiled_prediction,
)

from scripts.config_factory import pkl_load
from scripts.dataset_factory import build_pred_dataset
from scripts.io import npz_key, save_predictions_npz
from scripts.microsplit_factory import build_microsplit_module


def main(args: argparse.Namespace) -> Path:
    """Run end-to-end SW-tiled inference for a single experiment.

    Returns the path of the written NPZ file.
    """
    data_dir = args.data_root / args.dataset
    ckpt_path = args.ckpt_root / args.dataset / "BaselineVAECL_best.ckpt"
    pkl_path = args.ckpt_root / args.dataset / "config.pkl"
    save_dir = args.out_root / args.dataset / "predictions" / "sw_inner_tiling"

    # Derive STRIDE from target MMSE count + patch size (pkl) + overlap.
    pkl_data = pkl_load(pkl_path)["data"]
    is_3d = bool(pkl_data.get("mode_3D", False))
    img = int(pkl_data["image_size"])
    patch_size = [pkl_data["depth3D"], img, img] if is_3d else [img, img]
    if is_3d and args.stride_z is None:
        raise SystemExit(
            f"Dataset {args.dataset!r} is 3D but --stride-z was not provided."
        )
    stride, achieved = compute_stride_for_mmse_count(
        patch_size, args.overlap, args.mmse_count,
        stride_z=args.stride_z if is_3d else None,
    )
    print(
        f"patch_size={patch_size}, overlap={args.overlap}: "
        f"target MMSE count = {args.mmse_count}, achieved = {achieved}, "
        f"stride = {stride}"
    )

    dataset = build_pred_dataset(
        data_dir=data_dir,
        pkl_path=pkl_path,
        name=args.dataset,
        split=args.split,
        overlap=args.overlap,
        stride=stride,
        batch_size=args.batch_size,
        force_recompute_stats=args.force_recompute_stats,
    )
    print(f"dataset: n_patches={len(dataset)}, mode={dataset.config.mode}")

    loader = DataLoader(
        dataset,
        batch_size=dataset.config.batch_size,
        collate_fn=default_collate,
        num_workers=args.num_workers,
        shuffle=False,
    )

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else ("cuda" if args.device == "cuda" else "cpu")
    )
    print(f"device: {device}")
    # Per-tile model draws hardcoded to 1: canonical for SW, per-pixel coverage
    # comes from the patching geometry derived above.
    model = build_microsplit_module(
        ckpt_path=ckpt_path,
        pkl_path=pkl_path,
        mmse_count=1,
        device=device,
    )
    model.set_target_stats(
        dataset.normalization.target_means,
        dataset.normalization.target_stds,
    )

    preds_list, sources = sw_tiled_prediction(model, loader)

    results: dict[str, NDArray] = {}
    for data_idx, pred in enumerate(preds_list):
        source = sources[data_idx] if sources else "array"
        results[npz_key(source, data_idx)] = pred
    print(f"stitched {len(results)} image(s)")

    out_path = save_predictions_npz(results, save_dir)
    print(f"wrote {len(results)} prediction(s) to {out_path}")
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="microsplit-sw-tiled-predict",
        description=(
            "Run MicroSplit sliding-window tiled inference on one experiment, "
            "saving the per-image stitched predictions as a single .npz. "
            "Stride is derived from --mmse-count + tile geometry."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset", required=True,
        help="experiment name; resolves <data_root>/<dataset>/ and "
             "<ckpt_root>/<dataset>/",
    )
    p.add_argument(
        "--split", default="test", choices=["train", "val", "test"],
        help="which on-disk split to predict on",
    )
    p.add_argument(
        "--data-root", type=Path,
        default=Path("/project/careamics/switi/data"),
        help="root of <dataset>/{inputs,targets}/{train,val,test}/*.tif",
    )
    p.add_argument(
        "--ckpt-root", type=Path,
        default=Path("/project/careamics/switi/ckpts"),
        help="root of <dataset>/{BaselineVAECL_best.ckpt, config.pkl}",
    )
    p.add_argument(
        "--out-root", type=Path,
        default=Path("/project/careamics/switi/results"),
        help="root for predictions; output written to "
             "<out_root>/<dataset>/predictions/sw_inner_tiling/predictions.npz",
    )
    p.add_argument(
        "--overlap", type=int, nargs="+", default=[32, 32],
        metavar="N",
        help="tile overlap per spatial axis (length 2 for 2D, 3 for 3D)",
    )
    p.add_argument(
        "--mmse-count", type=int, default=64,
        help="target effective per-pixel coverage",
    )
    p.add_argument(
        "--stride-z", type=int, default=None,
        help="Z stride; required for 3D experiments, ignored for 2D",
    )
    p.add_argument(
        "--batch-size", type=int, default=128,
        help="prediction batch size",
    )
    p.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader num_workers",
    )
    p.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"],
        help='"auto" picks cuda when available, falls back to cpu',
    )
    p.add_argument(
        "--force-recompute-stats", action="store_true",
        help="bypass the <data_dir>/stats.json cache",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
