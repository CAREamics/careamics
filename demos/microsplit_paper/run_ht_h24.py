"""HT-H24 3D.

    python run_ht_h24.py                                   # all four stages
    python run_ht_h24.py --skip-n2v --skip-nm --skip-train # predict default/ckpts
    python run_ht_h24.py --smoke --experiment-name smoke   # every stage starts

Outputs: /project/careamics/experiments/paper/msplit/ht_h24/default/runs/<experiment-name>/{n2v,noise_models,train,predict}
(<experiment-name> defaults to e2e).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import tifffile
from careamics.lvae_training.dataset.utils.data_utils import get_datasplit_tuples

import msplit_common as P

HERE = P.PAPER_ROOT / "ht_h24"
CONDITIONS = {"default": HERE / "default"}

AXES = "SCZYX"
PATCH_SIZE = (9, 64, 64)
STRIDES = [1, 2, 2]
MULTISCALE_COUNT = 1
OUTPUT_CHANNELS = 2
BATCH_SIZE = 32
DENOISPLIT_WEIGHT = 0.9
LR = 1e-3
MAX_EPOCHS = 208
AUGMENTATIONS: list[str] = []
GRID_Z = None
PREDICT_FROM = "last"

N2V_EPOCHS = 10
N2V_PATCH_SIZE = (8, 64, 64)
N2V_TILE_SIZE = (8, 128, 128)
N2V_TILE_OVERLAP = (0, 32, 32)
N2V_PICK = "last"
NM = dict(n_gaussian=3, n_coeff=3, min_sigma=200.0, n_epochs=2000, domain="signal")

DATA_FILE = (
    "/group/jug/public_html/microsplit/ht_h24_3d_tiff/WTC11_WT_DIV25_3_1_0001.tif"
)
Z_START, Z_STOP = 25, 40
CHANNELS = [0, 1]
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1


def load_split_arrays():
    """(train, val, test) x (input, target) SCZYX float32 arrays."""
    volume = tifffile.imread(DATA_FILE)
    if volume.ndim != 5:
        raise ValueError(f"expected (T, Z, C, Y, X), got {volume.shape}")
    target = np.moveaxis(volume[:, Z_START:Z_STOP, CHANNELS].astype(np.float32), 2, 1)
    del volume
    tr, va, te = get_datasplit_tuples(VAL_FRACTION, TEST_FRACTION, target.shape[0])
    order = np.random.RandomState(955).permutation(target.shape[0])
    tr, va, te = order[tr], order[va], order[te]
    inp = (0.5 * target[:, 0:1] + 0.5 * target[:, 1:2]).astype(target.dtype)
    print(f"[DATA] {target.shape}: train {tr} val {va} test {te}", flush=True)
    return inp[tr], target[tr], inp[va], target[va], inp[te], target[te]


def flatten_z(stack: np.ndarray) -> np.ndarray:
    """(S, Z, Y, X, C) -> (S*Z, Y, X, C): the bundled run scored per 2D slice."""
    s, z, y, x, c = stack.shape
    return stack.reshape(s * z, y, x, c)


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
        nm_input = [np.moveaxis(tr_tg, 1, -1)]
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
        max_steps=args.max_steps,
        lr=LR,
        encoder_conv_strides=STRIDES,
        decoder_conv_strides=STRIDES,
        augmentations=AUGMENTATIONS,
        devices=args.devices,
        wandb=args.wandb,
        smoke=args.smoke,
    )
    careamist = P.make_careamist(config, run.train)
    if training:
        P.attach_noise_model(careamist, nm_paths)
        dm = P.make_datamodule(careamist, [tr_in], [tr_tg], [va_in], [va_tg])
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
            P.compute_normalization(careamist, [tr_in], [tr_tg], [va_in], [va_tg])

    if args.skip_predict:
        return
    ev_in, ev_tg = (te_in, te_tg) if args.eval_split == "test" else (va_in, va_tg)
    ev_in, ev_tg = [ev_in], [ev_tg]
    mmse = P.SMOKE_MMSE if args.smoke else args.mmse
    if args.smoke:
        ev_in, ev_tg = P.smoke_crop(ev_in, z=10), P.smoke_crop(ev_tg, z=10)
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
        [flatten_z(p) for p in preds],
        [flatten_z(np.moveaxis(t, 1, -1)) for t in ev_tg],
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
