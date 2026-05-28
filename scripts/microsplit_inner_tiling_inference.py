"""MicroSplit — classical tiled inference (CL-args entry point).

Headless companion to :mod:`scripts.microsplit_tiled_inference`. Loads a
pre-trained MicroSplit checkpoint, runs prediction with classical inner tiling
(`TiledPatching`, non-overlapping kept regions), stitches via the canonical
`convert_prediction(..., tiled=True)` path, and saves a single `.npz` keyed by
input-image identifier.

Run from the repo root:

    python -m scripts.microsplit_tiled_predict --dataset HT_LIF24_5ms
    python -m scripts.microsplit_tiled_predict --dataset CARE3D_liver \\
        --overlap 0 32 32 --mmse-count 50 --batch-size 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from careamics.dataset.image_region_data import ImageRegionData
from careamics.dev.sliding_window_tiled_pred import _move_input_to_device
from careamics.lightning.prediction.convert_prediction import convert_prediction

from scripts.dataset_factory import build_pred_dataset
from scripts.io import npz_key, save_predictions_npz
from scripts.microsplit_factory import build_microsplit_module


def main(args: argparse.Namespace) -> Path:
    """Run end-to-end tiled inference for a single experiment.

    Returns the path of the written NPZ file.
    """
    data_dir = args.data_root / args.dataset
    ckpt_path = args.ckpt_root / args.dataset / "BaselineVAECL_best.ckpt"
    pkl_path = args.ckpt_root / args.dataset / "config.pkl"
    save_dir = args.out_root / args.dataset / "predictions" / "inner_tiling"

    dataset = build_pred_dataset(
        data_dir=data_dir,
        pkl_path=pkl_path,
        name=args.dataset,
        split=args.split,
        overlap=args.overlap,
        stride=None,
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
    model = build_microsplit_module(
        ckpt_path=ckpt_path,
        pkl_path=pkl_path,
        mmse_count=args.mmse_count,
        device=device,
    )
    model.set_target_stats(
        dataset.normalization.target_means,
        dataset.normalization.target_stds,
    )

    # Collect all batched mean regions; `convert_prediction(tiled=True)` then
    # decollates, groups by `data_idx`, and stitches each image in one shot via
    # `stitch_single_prediction` (direct paste — correct for non-overlapping
    # kept regions).
    predictions: list[ImageRegionData] = []
    with torch.inference_mode():
        for batch_idx, batch in tqdm(
            enumerate(loader), total=len(loader), desc="Predicting"
        ):
            batch = _move_input_to_device(batch, device)
            mean_region_batch, _std = model.predict_step(batch, batch_idx)
            predictions.append(mean_region_batch)

    preds_list, sources = convert_prediction(
        predictions, tiled=True, restore_shape=False
    )
    print(f"stitched {len(preds_list)} image(s)")

    results: dict[str, NDArray] = {}
    for data_idx, pred in enumerate(preds_list):
        source = sources[data_idx] if sources else "array"
        results[npz_key(source, data_idx)] = pred

    out_path = save_predictions_npz(results, save_dir)
    print(f"wrote {len(results)} prediction(s) to {out_path}")
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="microsplit-tiled-predict",
        description=(
            "Run MicroSplit classical tiled inference on one experiment, "
            "saving the per-image stitched predictions as a single .npz."
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
             "<out_root>/<dataset>/predictions/inner_tiling/predictions.npz",
    )
    p.add_argument(
        "--overlap", type=int, nargs="+", default=[32, 32],
        metavar="N",
        help="tile overlap per spatial axis (length 2 for 2D, 3 for 3D)",
    )
    p.add_argument(
        "--mmse-count", type=int, default=50,
        help="number of stochastic forward passes per tile "
             "(per-tile MMSE; ~50 is canonical for classical inner tiling)",
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
