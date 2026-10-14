"""Pavia-P24 (Table 1 rows VI, VII), end to end: N2V -> noise models -> MicroSplit -> predict.

Conditions = the bundled runs (own noise models, 3 gaussians / 3 coeffs /
min_sigma 0.125 over the N2V signal range, from a 10-epoch N2V):

    high_highskew     HighSkew / High   (Cond_3/Main),       419 epochs, last ckpt,
                      test 25.08 / 23.94 dB
    medium_balanced   Balanced / Medium (Cond_1/Divided_2),  449 epochs, last ckpt,
                      test 22.79 / 23.74 dB
    (range-invariant PSNR, test split, mmse 50)

An epoch is the whole training set (no step cap, no early stopping). Both bundled
runs trained on 4 GPUs (DDP, 8 per GPU); the default here is 1 GPU at the same
global batch 32. Their N2V predates the N2V mask fix; a fresh run uses the fixed one.

    python run_pavia_p24.py --condition high_highskew                   # all stages
    python run_pavia_p24.py --condition medium_balanced --skip-n2v --skip-nm --skip-train
    python run_pavia_p24.py --condition high_highskew --smoke --experiment-name smoke

Outputs: /project/careamics/experiments/paper/msplit/pavia_p24/<condition>/runs/<experiment-name>/{n2v,noise_models,train,predict}.
Training resumes this experiment's newest last checkpoint.
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

HERE = P.PAPER_ROOT / "pavia_p24"  # bundle folder: ckpts, noise models, run outputs

# --- bundled runs pavia_high_highskew_419ep / pavia_medium_balanced_449ep ------
# (disentangle 2408/D24-M3-S0-L8/9 and /11 for the data and split)
CONDITIONS = {
    "high_highskew": dict(
        alpha_level="HighSkew", power_level="High", val=[2, 4], test=[1, 3], epochs=419
    ),
    "medium_balanced": dict(
        alpha_level="Balanced",
        power_level="Medium",
        val=[8, 14],
        test=[7, 13],
        epochs=449,
    ),
}
AXES = "SCYX"
PATCH_SIZE = (64, 64)
MULTISCALE_COUNT = 3
OUTPUT_CHANNELS = 2
BATCH_SIZE = 32
DENOISPLIT_WEIGHT = 0.9
PREDICT_FROM = "last"

N2V_EPOCHS = 10
N2V_PATCH_SIZE = (64, 64)
N2V_TILE_SIZE = (256, 256)
N2V_TILE_OVERLAP = (48, 48)
N2V_PICK = "last"
NM = dict(n_gaussian=3, n_coeff=3, min_sigma=0.125, n_epochs=2000, domain="signal")

# --- data (from scripts/microsplit_pavia_p24.py + conversion/ms_sources.py) ---
# Mirror: Cond_{1,2,3}/{Main,Divided_2,Divided_4}/*.tif, each (N, Y, X, C=2).
# pavia3_rawdata_loader: the subdir's non-json files, sorted by RELATIVE path
# (load-bearing: val/test indices are positional over the concatenation);
# explicit val/test frame indices per condition, read from each run's config.pkl;
# train = every other frame. Input = 0.5 ch0 + 0.5 ch1.
DATA_ROOT = "/group/jug/public_html/microsplit/pavia_p24_tiff"
ALPHA_SUBDIR = {"Balanced": "Cond_1", "MediumSkew": "Cond_2", "HighSkew": "Cond_3"}
POWER_SUBDIR = {"High": "Main", "Medium": "Divided_2", "Low": "Divided_4"}


def loader_files(alpha_level: str, power_level: str) -> list[str]:
    subdir = os.path.join(ALPHA_SUBDIR[alpha_level], POWER_SUBDIR[power_level])
    rels = []
    for fname in os.listdir(os.path.join(DATA_ROOT, subdir)):
        rel = os.path.join(subdir, fname)
        ext = fname.split(".")[-1]
        if ext == "json":
            continue
        if ext not in ("tif", "nd2"):
            raise ValueError(f"unexpected extension in {rel}")
        rels.append(rel)
    return sorted(rels)


def synthesize_input(target: np.ndarray) -> np.ndarray:
    return (0.5 * target[:, 0:1] + 0.5 * target[:, 1:2]).astype(target.dtype)


def load_split_arrays(cond: dict):
    """(train, val, test) x (input, target), lists of (1, C, Y, X) float32 frames."""
    rels = loader_files(cond["alpha_level"], cond["power_level"])
    per_file = [
        tifffile.imread(os.path.join(DATA_ROOT, r)).astype(np.float32) for r in rels
    ]
    index = [(fi, k) for fi, arr in enumerate(per_file) for k in range(arr.shape[0])]
    val, test = list(cond["val"]), list(cond["test"])
    bad = [i for i in val + test if not 0 <= i < len(index)]
    if bad:
        raise IndexError(f"val/test indices {bad} out of range for {len(index)} frames")
    train = [i for i in range(len(index)) if i not in val and i not in test]

    def take(idx: list[int]):
        tg = []
        for g in sorted(idx):
            fi, k = index[g]
            tg.append(np.moveaxis(per_file[fi][k : k + 1], -1, 1))  # (1, C, Y, X)
        return [synthesize_input(t) for t in tg], tg

    print(
        f"[DATA] {rels}: {len(index)} frames -> train {len(train)}, val {val}, "
        f"test {test}",
        flush=True,
    )
    return (*take(train), *take(val), *take(test))


def main(args: argparse.Namespace) -> None:
    P.check_args(args)
    cond = CONDITIONS[args.condition]
    cond_dir = HERE / args.condition
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
            **cond,
        },
    )
    tr_in, tr_tg, va_in, va_tg, te_in, te_tg = load_split_arrays(cond)

    # 1 + 2. noise models
    if args.skip_nm:
        nm_paths = (
            [Path(p) for p in args.nm_paths]
            if args.nm_paths
            else P.bundled_noise_models(cond_dir, OUTPUT_CHANNELS)
        )
    else:
        nm_input = [np.moveaxis(t, 1, -1) for t in tr_tg]  # (1, Y, X, C) frames
        if args.smoke:
            nm_input = P.smoke_crop_channel_last(nm_input, yx=256, n_stacks=2)
        n2v_ckpt = None
        if args.skip_n2v:
            n2v_ckpt = args.n2v_ckpt or P.single_file(cond_dir / "n2v_ckpt")
        n2v_pred = P.n2v_stage(
            nm_input,
            work_dir=run.n2v,
            name=run.experiment,
            axes="SYXC",
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

    # 3. MicroSplit
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
        max_epochs=args.num_epochs or cond["epochs"],
        max_steps=args.max_steps,
        devices=args.devices,
        wandb=args.wandb,
        smoke=args.smoke,
    )
    careamist = P.make_careamist(config, run.train)
    if training:
        P.attach_noise_model(careamist, nm_paths)
        dm = P.make_datamodule(careamist, tr_in, tr_tg, va_in, va_tg)
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

    # 4. predict + metrics
    if args.skip_predict:
        return
    ev_in, ev_tg = (te_in, te_tg) if args.eval_split == "test" else (va_in, va_tg)
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
    P.add_common_args(
        parser, conditions=list(CONDITIONS), default_condition="high_highskew"
    )
    main(parser.parse_args())
