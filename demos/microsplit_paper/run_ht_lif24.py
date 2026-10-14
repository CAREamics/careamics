"""HT-LIF24.

    python run_ht_lif24.py --condition 5ms                                  # all stages
    python run_ht_lif24.py --condition 5ms --skip-n2v --skip-nm --skip-train # predict 5ms/ckpts
    python run_ht_lif24.py --condition 3split_500ms --smoke --experiment-name smoke

Outputs: /project/careamics/experiments/paper/msplit/ht_lif24/<condition>/runs/<experiment-name>/{n2v,noise_models,train,predict}
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

HERE = P.PAPER_ROOT / "ht_lif24"

CONDITIONS = {
    "5ms": dict(
        exposure="5ms",
        channels=[0, 1, 8],
        denoisplit_weight=0.9,
        batch_size=64,
        early_stop_patience=None,
        predict_from="last",
        ckpt="best_0_1_careamist.ckpt",
        predict_batch=64,
    ),
    "3split_500ms": dict(
        exposure="500ms",
        channels=[1, 2, 3, 17],
        denoisplit_weight=0.0,
        batch_size=32,
        early_stop_patience=200,
        predict_from="best",
        ckpt="best_1_2_3.ckpt",
        predict_batch=64,
    ),
}
AXES = "SCYX"
PATCH_SIZE = (64, 64)
MULTISCALE_COUNT = 3
MAX_EPOCHS = 400

N2V_EPOCHS = 10
N2V_PATCH_SIZE = (64, 64)
N2V_TILE_SIZE = (256, 256)
N2V_TILE_OVERLAP = (48, 48)
N2V_PICK = "last"
NM = dict(n_gaussian=3, n_coeff=3, min_sigma=200.0, n_epochs=2000, domain="signal")

DATA_ROOT = "/group/jug/public_html/microsplit/ht_lif24_tiff"
HIGHSNR_EXPOSURE = "500ms"
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1


def load_exposure(exposure: str, channels: list[int]) -> np.ndarray:
    """All frames of Set1..Set6, channels selected -> (N, C, Y, X) float32."""
    return np.concatenate(
        [
            tifffile.imread(
                os.path.join(DATA_ROOT, exposure, f"Set{i}", f"uSplit_{exposure}.tif")
            )[:, channels]
            for i in range(1, 7)
        ],
        axis=0,
    ).astype(np.float32)


def splits(n: int):
    return get_datasplit_tuples(VAL_FRACTION, TEST_FRACTION, n)


def main(args: argparse.Namespace) -> None:
    P.check_args(args)
    cond = CONDITIONS[args.condition]
    cond_dir = HERE / args.condition
    run = P.RunDirs(cond_dir, args.experiment_name)
    batch = args.batch_size or cond["batch_size"]
    n_out = len(cond["channels"]) - 1
    use_nm = cond["denoisplit_weight"] > 0
    P.log_parameters(
        args,
        {
            "condition_dir": cond_dir,
            "run_dir": run.root,
            "PATCH_SIZE": PATCH_SIZE,
            "BATCH_SIZE": batch,
            "NM": NM,
            **cond,
        },
    )
    data = load_exposure(cond["exposure"], cond["channels"])
    tr, va, te = splits(len(data))
    tr_in, tr_tg = data[tr, -1:], data[tr, :-1]
    va_in, va_tg = data[va, -1:], data[va, :-1]
    print(
        f"[DATA] {cond['exposure']} {data.shape}: train {len(tr)} val {len(va)} "
        f"test {len(te)} frames",
        flush=True,
    )

    nm_paths = None
    if use_nm and args.skip_nm:
        nm_paths = (
            [Path(p) for p in args.nm_paths]
            if args.nm_paths
            else P.bundled_noise_models(cond_dir, n_out)
        )
    elif use_nm:
        nm_input = [np.moveaxis(data[tr], 1, -1)]
        if args.smoke:
            nm_input = P.smoke_crop_channel_last(nm_input, yx=256, n_samples=2)
        n2v_ckpt = None
        if args.skip_n2v:
            n2v_ckpt = args.n2v_ckpt or P.single_file(cond_dir / "n2v_ckpt")
        n2v_pred = P.n2v_stage(
            nm_input,
            work_dir=run.n2v,
            name=run.experiment,
            axes="SYXC",
            n_channels=len(cond["channels"]),
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
        nm_paths = P.fit_noise_models(
            [p[..., :-1] for p in n2v_pred],
            [x[..., :-1] for x in nm_input],
            run.nm,
            **nm_kw,
        )
    else:
        print("[NM] pure muSplit: no N2V, no noise models", flush=True)

    training = not args.skip_train
    config = P.microsplit_config(
        run.experiment,
        axes=AXES,
        patch_size=PATCH_SIZE,
        output_channels=n_out,
        multiscale_count=MULTISCALE_COUNT,
        batch_size=batch,
        num_workers=args.num_workers,
        seed=args.seed,
        denoisplit_weight=cond["denoisplit_weight"] if training else 0.0,
        max_epochs=args.num_epochs or MAX_EPOCHS,
        max_steps=args.max_steps,
        early_stop_patience=cond["early_stop_patience"],
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
            if cond["predict_from"] == "best"
            else P.latest_last_checkpoint(run.train, run.experiment)
        )
    else:
        ckpt = Path(args.ckpt) if args.ckpt else cond_dir / "ckpts" / cond["ckpt"]
        if not args.skip_predict:
            P.compute_normalization(careamist, [tr_in], [tr_tg], [va_in], [va_tg])

    if args.skip_predict:
        return
    idx = te if args.eval_split == "test" else va
    ev_in = [data[idx, -1:]]
    del data
    hs = load_exposure(HIGHSNR_EXPOSURE, cond["channels"])
    if len(hs) != len(tr) + len(va) + len(te):
        raise RuntimeError(
            f"{HIGHSNR_EXPOSURE} has {len(hs)} frames, the input "
            f"{len(tr) + len(va) + len(te)}"
        )
    ev_tg = [hs[idx, :-1]]
    del hs
    mmse = P.SMOKE_MMSE if args.smoke else args.mmse
    if args.smoke:
        ev_in, ev_tg = P.smoke_crop(ev_in), P.smoke_crop(ev_tg)
    preds = P.predict_stage(
        careamist,
        ev_in,
        ckpt=ckpt,
        out_dir=run.predict,
        patch_size=PATCH_SIZE,
        mmse=mmse,
        batch_size=cond["predict_batch"],
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
    P.add_common_args(parser, conditions=list(CONDITIONS), default_condition="5ms")
    main(parser.parse_args())
