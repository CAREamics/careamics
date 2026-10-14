"""CBG-N18 3D.

    python run_cbg_n18.py                                   # all four stages
    python run_cbg_n18.py --skip-n2v --skip-nm --skip-train # predict default/ckpts
    python run_cbg_n18.py --smoke --experiment-name smoke   # every stage starts

Outputs: /project/careamics/experiments/paper/msplit/cbg_n18/default/runs/<experiment-name>/{n2v,noise_models,train,predict}
(<experiment-name> defaults to e2e).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import tifffile

import msplit_common as P

HERE = P.PAPER_ROOT / "cbg_n18"
CONDITIONS = {"default": HERE / "default"}

AXES = "SCZYX"
PATCH_SIZE = (5, 64, 64)
STRIDES = [1, 2, 2]
MULTISCALE_COUNT = 1
OUTPUT_CHANNELS = 3
BATCH_SIZE = 16
DENOISPLIT_WEIGHT = 0.9
STEPS_PER_EPOCH = 2000
MAX_STEPS = 800_000
EARLY_STOP_PATIENCE = 200
MAX_EPOCHS = 400
GRID_Z = None
PREDICT_FROM = "best"

N2V_EPOCHS = 20
N2V_PATCH_SIZE = (4, 64, 64)
N2V_TILE_SIZE = (8, 256, 256)
N2V_TILE_OVERLAP = (0, 32, 32)
N2V_PICK = "best"
NM = dict(n_gaussian=6, n_coeff=4, min_sigma=0.125, n_epochs=2000, domain="union")

DATA_ROOT = "/group/jug/public_html/microsplit/cbg_n18_tiff"
LIVER_FILE = "channel_234_1.tif"
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1


def synthesize_input(target: np.ndarray) -> np.ndarray:
    return target.mean(axis=1, keepdims=True).astype(target.dtype)


def load_split_arrays():
    """(train, val, test) x (input, target), lists of (1, C, Z, Y, X) float32."""
    arr = tifffile.imread(os.path.join(DATA_ROOT, LIVER_FILE))
    if arr.ndim != 4 or arr.shape[0] != 3:
        raise ValueError(f"{LIVER_FILE}: expected (3, Z, Y, X), got {arr.shape}")
    n_z = arr.shape[1]
    test_end = int(TEST_FRACTION * n_z)
    val_end = test_end + int(VAL_FRACTION * n_z)
    cut = {"train": (val_end, n_z), "val": (test_end, val_end), "test": (0, test_end)}
    tg = {
        k: [np.ascontiguousarray(arr[:, a:b])[None].astype(np.float32)]
        for k, (a, b) in cut.items()
    }
    print(
        f"[DATA] train {[t.shape for t in tg['train']]} val {[t.shape for t in tg['val']]} "
        f"test {[t.shape for t in tg['test']]}",
        flush=True,
    )
    out = []
    for split in ("train", "val", "test"):
        out += [[synthesize_input(t) for t in tg[split]], tg[split]]
    return tuple(out)


def main(args: argparse.Namespace) -> None:
    P.check_args(args)
    cond_dir = CONDITIONS[args.condition]
    run = P.RunDirs(cond_dir, args.experiment_name)
    batch = args.batch_size or BATCH_SIZE
    P.log_parameters(
        args,
        {
            "condition_dir": cond_dir,
            "run_dir": run.root,
            "PATCH_SIZE": PATCH_SIZE,
            "BATCH_SIZE": batch,
            "NM": NM,
        },
    )
    tr_in, tr_tg, va_in, va_tg, te_in, te_tg = load_split_arrays()

    if args.skip_nm:
        nm_paths = (
            [Path(p) for p in args.nm_paths]
            if args.nm_paths
            else P.bundled_noise_models(cond_dir, OUTPUT_CHANNELS)
        )
    else:
        nm_input = [np.moveaxis(t, 1, -1) for t in tr_tg]
        if args.smoke:
            nm_input = P.smoke_crop_channel_last(nm_input, yx=256, z=8)
        n2v_ckpt = None
        if args.skip_n2v:
            n2v_ckpt = args.n2v_ckpt or P.single_file(cond_dir / "n2v_ckpt")
        n2v_pred = P.n2v_stage(
            nm_input,
            work_dir=run.n2v,
            name=run.experiment,
            axes="SZYXC",
            n_channels=OUTPUT_CHANNELS,
            patch_size=N2V_PATCH_SIZE,
            epochs=args.n2v_epochs or N2V_EPOCHS,
            batch_size=batch,
            seed=args.seed,
            tile_size=N2V_TILE_SIZE,
            tile_overlap=N2V_TILE_OVERLAP,
            pick=N2V_PICK,
            n2v_ckpt=n2v_ckpt,
            smoke=args.smoke,
        )
        nm_kw = dict(NM, n_epochs=P.SMOKE_NM_EPOCHS) if args.smoke else NM
        nm_paths = P.fit_noise_models(n2v_pred, nm_input, run.nm, **nm_kw)

    training = not args.skip_train
    config = P.microsplit_config(
        run.experiment,
        axes=AXES,
        patch_size=PATCH_SIZE,
        output_channels=OUTPUT_CHANNELS,
        multiscale_count=MULTISCALE_COUNT,
        batch_size=batch,
        num_workers=args.num_workers,
        seed=args.seed,
        denoisplit_weight=DENOISPLIT_WEIGHT if training else 0.0,
        max_epochs=args.num_epochs or MAX_EPOCHS,
        max_steps=args.max_steps or MAX_STEPS,
        steps_per_epoch=STEPS_PER_EPOCH,
        early_stop_patience=EARLY_STOP_PATIENCE,
        encoder_conv_strides=STRIDES,
        decoder_conv_strides=STRIDES,
        devices=args.devices,
        wandb=args.wandb,
        smoke=args.smoke,
    )
    careamist = P.make_careamist(config, run.train)
    if training:
        P.attach_noise_model(careamist, nm_paths)
        dm = P.make_datamodule(
            careamist,
            tr_in,
            tr_tg,
            va_in,
            va_tg,
            samples_per_epoch=STEPS_PER_EPOCH * batch,
        )
        best = P.train_stage(
            careamist, dm, work_dir=run.train, resume=not args.no_resume
        )
        ckpt = (
            best
            if PREDICT_FROM == "best"
            else P.latest_last_checkpoint(run.train, run.experiment)
        )
    else:
        ckpt = Path(args.ckpt) if args.ckpt else P.single_file(cond_dir / "ckpts")
        if not args.skip_predict:
            P.compute_normalization(careamist, tr_in, tr_tg, va_in, va_tg)

    if args.skip_predict:
        return
    ev_in, ev_tg = (te_in, te_tg) if args.eval_split == "test" else (va_in, va_tg)
    mmse = P.SMOKE_MMSE if args.smoke else args.mmse
    if args.smoke:
        ev_in, ev_tg = P.smoke_crop(ev_in, z=8), P.smoke_crop(ev_tg, z=8)
    preds = P.predict_stage(
        careamist,
        ev_in,
        ckpt=ckpt,
        out_dir=run.predict,
        patch_size=PATCH_SIZE,
        grid_z=GRID_Z,
        mmse=mmse,
        batch_size=batch,
        seed=args.seed,
    )
    P.evaluate(
        preds,
        [np.moveaxis(t, 1, -1) for t in ev_tg],
        run.predict,
        tag=f"{args.eval_split}_mmse{mmse}",
        save=not args.smoke,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    P.add_common_args(parser, conditions=list(CONDITIONS), default_condition="default")
    main(parser.parse_args())
