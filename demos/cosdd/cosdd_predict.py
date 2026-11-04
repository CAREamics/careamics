"""Denoise the test set with a trained COSDD model and compute PSNR.

Estimates: the direct denoiser, one random sample, and the MMSE over each
`--n-samples` (the paper reports MMSE over 100 samples). Whole frames are
predicted, without tiling, as in the reference.

PSNR: skimage PSNR per frame with data_range = max - min of the ground-truth
frame, averaged over frames (see `cosdd_data` for the ground truth of each
dataset). Range-invariant PSNR (careamics `scale_invariant_psnr`) is reported
next to it.

Example:
    python cosdd_predict.py --run-dir /group/jug/Igor/cosdd/runs/cosdd_conva_large
    python cosdd_predict.py --run-dir /group/jug/Igor/cosdd/runs/smoke --smoke
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tifffile
from cosdd_data import PAPER_PSNR, load
from numpy.typing import NDArray

from careamics import CAREamist
from careamics.metrics.metrics import psnr, scale_invariant_psnr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--checkpoint",
        default="last",
        help="'last' (as the reference), 'best' (lowest val_loss) or a path",
    )
    p.add_argument("--n-samples", type=int, nargs="+", default=[10, 100])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="first 4 test frames only")
    return p.parse_args()


def resolve_checkpoint(run_dir: Path, which: str) -> Path:
    if which in ("last", "best"):
        info = json.loads((run_dir / "run_info.json").read_text())
        return Path(info[f"{which}_model_path"])
    return Path(which)


def frame_psnrs(gt: NDArray, pred: NDArray) -> tuple[float, float]:
    plain = [
        psnr(g, p, data_range=float(g.max() - g.min()))
        for g, p in zip(gt, pred, strict=True)
    ]
    rinv = [float(scale_invariant_psnr(g, p)) for g, p in zip(gt, pred, strict=True)]
    return float(np.mean(plain)), float(np.mean(rinv))


def main() -> None:
    args = parse_args()
    ckpt = resolve_checkpoint(args.run_dir, args.checkpoint)
    info = json.loads((args.run_dir / "run_info.json").read_text())
    data = load(info["dataset"])
    test, gt = data.test, data.gt
    if args.smoke:
        test, gt = test[:4], gt[:4]
    print(f"[DATA] {data.name}: test {test.shape}, gt {gt.shape}")
    print(f"[CKPT] {ckpt}")

    careamist = CAREamist(checkpoint_path=ckpt, work_dir=args.run_dir)
    model = careamist.model
    predict_kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers}

    tag = args.checkpoint if args.checkpoint in ("last", "best") else ckpt.stem
    out_dir = args.run_dir / f"predict_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    estimates: dict[str, NDArray] = {}
    timings: dict[str, float] = {}

    if model.direct_denoiser is not None:
        t0 = time.time()
        model.direct_pred = True
        pred, _ = careamist.predict(test, **predict_kwargs)
        estimates["direct"] = pred[0]
        timings["direct"] = time.time() - t0

    # a single random sample: sampling path with n_samples = 1
    t0 = time.time()
    model.direct_pred = False
    pred, _ = careamist.predict(test, **predict_kwargs)
    estimates["sample"] = pred[0]
    timings["sample"] = time.time() - t0

    for n in args.n_samples:
        t0 = time.time()
        pred, std, _ = careamist.predict(test, n_predictions=n, **predict_kwargs)
        estimates[f"mmse{n}"] = pred[0]
        timings[f"mmse{n}"] = time.time() - t0
        if n == max(args.n_samples):
            tifffile.imwrite(out_dir / f"mmse{n}_std.tif", std[0].astype(np.float32))
    model.direct_pred = model.direct_denoiser is not None

    metrics = {}
    paper = PAPER_PSNR[data.name]
    print(f"\n[PSNR] {data.name}, {len(test)} frames, checkpoint {ckpt.name}")
    print(f"{'estimate':>10} {'PSNR':>8} {'rinvPSNR':>9} {'paper':>7} {'min':>7}")
    for name, est in estimates.items():
        tifffile.imwrite(out_dir / f"{name}.tif", est.astype(np.float32))
        plain, rinv = frame_psnrs(gt, est)
        metrics[name] = {
            "psnr": plain,
            "rinv_psnr": rinv,
            "paper_psnr": paper.get(name),
            "minutes": round(timings[name] / 60, 2),
        }
        ref = f"{paper[name]:7.2f}" if name in paper else f"{'-':>7}"
        print(f"{name:>10} {plain:8.2f} {rinv:9.2f} {ref} {timings[name] / 60:7.1f}")
    plain, rinv = frame_psnrs(gt, test)
    metrics["noisy"] = {"psnr": plain, "rinv_psnr": rinv}
    print(f"{'noisy':>10} {plain:8.2f} {rinv:9.2f}")

    result = {
        "dataset": data.name,
        "checkpoint": str(ckpt),
        "n_test_frames": len(test),
        "smoke": args.smoke,
        "metrics": metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print(f"[OUT] {out_dir}")


if __name__ == "__main__":
    main()
