"""HT-P23B (Table 1 row III), end to end: N2V -> noise models -> MicroSplit -> predict.

Defaults reproduce the bundled run in `default/` (own noise models, fixed N2V):
`p23b_n2vfix`, best checkpoint epoch 384, test 26.51 / 21.62 dB range-invariant
PSNR at mmse 50, eval grid (1, 32, 32).

    python run_ht_p23b.py                                   # all four stages
    python run_ht_p23b.py --skip-n2v --skip-nm --skip-train # predict default/ckpts
    python run_ht_p23b.py --skip-n2v                        # refit NMs from default/n2v_ckpt
    python run_ht_p23b.py --smoke --experiment-name smoke   # every stage starts

Outputs: /project/careamics/experiments/paper/msplit/ht_p23b/default/runs/<experiment-name>/{n2v,noise_models,train,predict}.
Training resumes this experiment's newest last checkpoint, so a long run is the
same command submitted several times.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import msplit_common as P  # noqa: E402

HERE = P.PAPER_ROOT / "ht_p23b"   # bundle folder: ckpts, noise models, run outputs
CONDITIONS = {"default": HERE / "default"}

# --- bundled run p23b_n2vfix (disentangle 2408/D19-M3-S0-L8/11) ---------------
AXES = "SCZYX"
PATCH_SIZE = (5, 64, 64)            # depth3D 5
STRIDES = [1, 2, 2]                 # no Z downsampling
MULTISCALE_COUNT = 1
OUTPUT_CHANNELS = 2
BATCH_SIZE = 8
DENOISPLIT_WEIGHT = 0.9             # usplit 0.1
STEPS_PER_EPOCH = 2000              # limit_train_batches; dataset < 2000 x 8 -> A4
MAX_STEPS = 800_000
EARLY_STOP_PATIENCE = 200
MAX_EPOCHS = 400
GRID_Z = 1                          # paper_stats Test_P64_G32 -> (1, 32, 32)
PREDICT_FROM = "best"

N2V_EPOCHS = 20
N2V_PATCH_SIZE = (4, 64, 64)        # N2V needs even depth
N2V_TILE_SIZE = (8, 256, 256)
N2V_TILE_OVERLAP = (0, 32, 32)
N2V_PICK = "best"
NM = dict(n_gaussian=6, n_coeff=4, min_sigma=0.125, n_epochs=2000, domain="union")

# --- data (from scripts/microsplit_p23b.py) -----------------------------------
# Mirror: <root>/Experiment-NNN.tif (H=4, C, Z, Y, X), czifile's HCZYX.
# disentangle get_train_val_data for ExpMicroscopyV2: train 447, 449;
# val = 448 Z planes [0, Z//2), test = 448 Z planes [Z//2, Z). H index 3,
# channels [0, 2]; input = channel mean (alpha 1/K).
DATA_ROOT = "/group/jug/public_html/microsplit/ht_p23b_tiff"
TRAIN_FILES = ["Experiment-447.tif", "Experiment-449.tif"]
VALTEST_FILE = "Experiment-448.tif"
H_INDEX = 3
TARGET_CHANNELS = [0, 2]


def _load_one(rel: str) -> np.ndarray:
    data = tifffile.imread(os.path.join(DATA_ROOT, rel))  # (H, C, Z, Y, X)
    if data.ndim != 5:
        raise ValueError(f"{rel}: expected 5-D HCZYX, got {data.shape}")
    return np.take(data[H_INDEX], TARGET_CHANNELS, axis=0)[None].astype(np.float32)


def synthesize_input(target: np.ndarray) -> np.ndarray:
    return target.mean(axis=1, keepdims=True).astype(target.dtype)


def load_split_arrays():
    """(train, val, test) x (input, target), lists of (1, C, Z, Y, X) float32."""
    with open(os.path.join(DATA_ROOT, "metadata.json")) as fh:
        known = {e["rel"] for e in json.load(fh)["files"]}
    for rel in TRAIN_FILES + [VALTEST_FILE]:
        if rel not in known:
            raise FileNotFoundError(f"{rel} not in {DATA_ROOT}/metadata.json")
    tr_tg = [_load_one(rel) for rel in TRAIN_FILES]
    vt = _load_one(VALTEST_FILE)
    z_half = vt.shape[2] // 2
    va_tg = [np.ascontiguousarray(vt[:, :, :z_half])]
    te_tg = [np.ascontiguousarray(vt[:, :, z_half:])]
    print(f"[DATA] train {[t.shape for t in tr_tg]} val {[t.shape for t in va_tg]} "
          f"test {[t.shape for t in te_tg]}", flush=True)
    return ([synthesize_input(t) for t in tr_tg], tr_tg,
            [synthesize_input(t) for t in va_tg], va_tg,
            [synthesize_input(t) for t in te_tg], te_tg)


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
