"""HHMI-D25 8bit 3D (Table 1 row XXIII), end to end: N2V -> noise models -> MicroSplit -> predict.

Defaults reproduce the bundled run in `default/` (own noise models):
`hhmi_d25_paper`, best checkpoint epoch 257, test 22.75 / 31.50 / 24.53 dB
range-invariant PSNR at mmse 50, Z tiles abutting (Z grid 5). Its noise models
were fit from an N2V trained before the N2V mask fix; a fresh run uses the fixed one.

    python run_hhmi_d25.py                                   # all four stages
    python run_hhmi_d25.py --skip-n2v --skip-nm --skip-train # predict default/ckpts
    python run_hhmi_d25.py --smoke --experiment-name smoke   # every stage starts

Outputs: /project/careamics/experiments/paper/msplit/hhmi_d25/default/runs/<experiment-name>/{n2v,noise_models,train,predict}.
Training resumes this experiment's newest last checkpoint, so a long run is the
same command submitted several times.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import msplit_common as P  # noqa: E402

HERE = P.PAPER_ROOT / "hhmi_d25"   # bundle folder: ckpts, noise models, run outputs
CONDITIONS = {"default": HERE / "default"}

# --- bundled run hhmi_d25_paper (disentangle 2507/D32-M3-S0-L8/0) --------------
AXES = "SCZYX"
PATCH_SIZE = (5, 64, 64)            # depth3D 5
STRIDES = [1, 2, 2]
MULTISCALE_COUNT = 1
OUTPUT_CHANNELS = 3
BATCH_SIZE = 16
DENOISPLIT_WEIGHT = 0.9
STEPS_PER_EPOCH = 2000              # dataset < 2000 x 16 -> A4
MAX_STEPS = 800_000
EARLY_STOP_PATIENCE = 200
MAX_EPOCHS = 400
GRID_Z = None                       # bundled predict: Z tiles abut
PREDICT_FROM = "best"

N2V_EPOCHS = 10
N2V_PATCH_SIZE = (4, 64, 64)
N2V_TILE_SIZE = (8, 256, 256)
N2V_TILE_OVERLAP = (0, 32, 32)
N2V_PICK = "best"
NM = dict(n_gaussian=6, n_coeff=4, min_sigma=0.125, n_epochs=2000, domain="union")

# --- data (from scripts/microsplit_hhmi_d25.py) -------------------------------
# Mirror: Composite_region*.tif, (Z=51, C=6, 2048, 2048) uint8, ZCYX.
# disentangle hhmi_raw_dloader (run 2507/D32-M3-S0-L8/0): explicit files per
# split, channel_idx_list (1, 3, 2) -> 3 targets in that order; input = channel
# mean (alpha 1/K).
DATA_ROOT = "/group/jug/public_html/microsplit/hhmi_d25_tiff"
SPLIT_FILES = {
    "train": [
        "Composite_region3_6am_CF_L3L2.tif",
        "Composite_region5_6am_L3L1.tif",
        "Composite_region2_6AM_CF_L3L3.tif",
        "Composite_region4_6am_L3L2.tif",
    ],
    "val": ["Composite_region1_CF_L3L3.tif"],
    "test": ["Composite_region7_CF_12noon_L3L4.tif"],
}
CHANNELS = [1, 3, 2]


def _load_one(rel: str) -> np.ndarray:
    arr = tifffile.imread(os.path.join(DATA_ROOT, rel))      # (Z, C, Y, X)
    if arr.ndim != 4:
        raise ValueError(f"{rel}: expected ZCYX, got {arr.shape}")
    vol = np.take(arr, CHANNELS, axis=1)                     # (Z, C', Y, X)
    return np.moveaxis(vol, 1, 0)[None].astype(np.float32)   # (1, C', Z, Y, X)


def synthesize_input(target: np.ndarray) -> np.ndarray:
    return target.mean(axis=1, keepdims=True).astype(target.dtype)


def load_split_arrays():
    """(train, val, test) x (input, target), lists of (1, C, Z, Y, X) float32."""
    tg = {split: [_load_one(f) for f in files] for split, files in SPLIT_FILES.items()}
    print(f"[DATA] train {[t.shape for t in tg['train']]} val {[t.shape for t in tg['val']]} "
          f"test {[t.shape for t in tg['test']]}", flush=True)
    out = []
    for split in ("train", "val", "test"):
        out += [[synthesize_input(t) for t in tg[split]], tg[split]]
    return tuple(out)


def main(args: argparse.Namespace) -> None:
    P.check_args(args)
    cond_dir = CONDITIONS[args.condition]
    run = P.RunDirs(cond_dir, args.experiment_name)
    batch = args.batch_size or BATCH_SIZE
    P.log_parameters(args, {"condition_dir": cond_dir, "run_dir": run.root,
                            "PATCH_SIZE": PATCH_SIZE, "BATCH_SIZE": batch, "NM": NM})
    tr_in, tr_tg, va_in, va_tg, te_in, te_tg = load_split_arrays()

    # 1 + 2. noise models
    if args.skip_nm:
        nm_paths = ([Path(p) for p in args.nm_paths] if args.nm_paths
                    else P.bundled_noise_models(cond_dir, OUTPUT_CHANNELS))
    else:
        nm_input = [np.moveaxis(t, 1, -1) for t in tr_tg]          # (1, Z, Y, X, C)
        if args.smoke:
            nm_input = P.smoke_crop_channel_last(nm_input, yx=256, z=8)
        n2v_ckpt = None
        if args.skip_n2v:
            n2v_ckpt = args.n2v_ckpt or P.single_file(cond_dir / "n2v_ckpt")
        n2v_pred = P.n2v_stage(
            nm_input, work_dir=run.n2v, name=run.experiment, axes="SZYXC",
            n_channels=OUTPUT_CHANNELS, patch_size=N2V_PATCH_SIZE,
            epochs=args.n2v_epochs or N2V_EPOCHS, batch_size=batch, seed=args.seed,
            tile_size=N2V_TILE_SIZE, tile_overlap=N2V_TILE_OVERLAP, pick=N2V_PICK,
            n2v_ckpt=n2v_ckpt, smoke=args.smoke,
        )
        nm_kw = dict(NM, n_epochs=P.SMOKE_NM_EPOCHS) if args.smoke else NM
        nm_paths = P.fit_noise_models(n2v_pred, nm_input, run.nm, **nm_kw)

    # 3. MicroSplit
    training = not args.skip_train
    config = P.microsplit_config(
        run.experiment, axes=AXES, patch_size=PATCH_SIZE,
        output_channels=OUTPUT_CHANNELS, multiscale_count=MULTISCALE_COUNT,
        batch_size=batch, num_workers=args.num_workers, seed=args.seed,
        denoisplit_weight=DENOISPLIT_WEIGHT if training else 0.0,
        max_epochs=args.num_epochs or MAX_EPOCHS, max_steps=args.max_steps or MAX_STEPS,
        steps_per_epoch=STEPS_PER_EPOCH, early_stop_patience=EARLY_STOP_PATIENCE,
        encoder_conv_strides=STRIDES, decoder_conv_strides=STRIDES,
        devices=args.devices, wandb=args.wandb, smoke=args.smoke,
    )
    careamist = P.make_careamist(config, run.train)
    if training:
        P.attach_noise_model(careamist, nm_paths)
        dm = P.make_datamodule(careamist, tr_in, tr_tg, va_in, va_tg,
                               samples_per_epoch=STEPS_PER_EPOCH * batch)
        best = P.train_stage(careamist, dm, work_dir=run.train, resume=not args.no_resume)
        ckpt = best if PREDICT_FROM == "best" else P.latest_last_checkpoint(
            run.train, run.experiment)
    else:
        ckpt = Path(args.ckpt) if args.ckpt else P.single_file(cond_dir / "ckpts")
        if not args.skip_predict:
            P.compute_normalization(careamist, tr_in, tr_tg, va_in, va_tg)

    # 4. predict + metrics
    if args.skip_predict:
        return
    ev_in, ev_tg = (te_in, te_tg) if args.eval_split == "test" else (va_in, va_tg)
    mmse = P.SMOKE_MMSE if args.smoke else args.mmse
    if args.smoke:
        ev_in, ev_tg = P.smoke_crop(ev_in, z=8), P.smoke_crop(ev_tg, z=8)
    preds = P.predict_stage(careamist, ev_in, ckpt=ckpt, out_dir=run.predict,
                            patch_size=PATCH_SIZE, grid_z=GRID_Z, mmse=mmse,
                            batch_size=batch, seed=args.seed)
    P.evaluate(preds, [np.moveaxis(t, 1, -1) for t in ev_tg], run.predict,
               tag=f"{args.eval_split}_mmse{mmse}", save=not args.smoke)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    P.add_common_args(parser, conditions=list(CONDITIONS), default_condition="default")
    main(parser.parse_args())
