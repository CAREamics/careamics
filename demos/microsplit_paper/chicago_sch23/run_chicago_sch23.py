"""Chicago-Sch23 (Table 1 rows XV, XX, XXX), end to end: MicroSplit -> predict.

Pure muSplit, as the paper runs: no noise model, so stages 1-2 (N2V, noise
models) do nothing. Conditions = the bundled runs:

    4ch   chicago4_paper           channels 0-3, best ckpt epoch 393,
                                   test 35.46 / 39.46 / 38.59 / 36.06 dB
    ch01  chicago_ch01_32ep        channels 0, 1, last ckpt epoch 31 (32 epochs),
                                   test 40.42 / 41.98 dB
    ch23  chicago_ch23_paper ->    channels 2, 3: paper schedule (380k steps),
          chicago_ch23_lrdecay     then a StepLR continuation warm-started from its
                                   last ckpt (lr 5e-4, x0.5 / 30 epochs, 120k steps);
                                   best ckpt epoch 42, test 66.57 / 43.67 dB
    (range-invariant PSNR, test split, mmse 50)

    python run_chicago_sch23.py --condition ch23                       # train + predict
    python run_chicago_sch23.py --condition 4ch --skip-train           # predict 4ch/ckpts
    python run_chicago_sch23.py --condition ch01 --smoke --experiment-name smoke

The bundled runs trained on 4 GPUs (DDP, 8 per GPU) except the ch23 continuation;
the default here is 1 GPU at the same global batch 32. Outputs:
/project/careamics/experiments/paper/msplit/chicago_sch23/<condition>/runs/<experiment-name>/{train,predict}. Training resumes each phase's
newest last checkpoint, so a long run is the same command submitted several times.
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
from careamics.lvae_training.dataset.utils.data_utils import (
    get_datasplit_tuples,
)  # noqa: E402

HERE = P.PAPER_ROOT / "chicago_sch23"  # bundle folder: ckpts, noise models, run outputs

# --- bundled runs (disentangle 2406/D18-M3-S0-L8/0, 2501/D18-M3-S0-L8/1, /3) ----
AXES = "SCYX"
PATCH_SIZE = (64, 64)
MULTISCALE_COUNT = 3
BATCH_SIZE = 32
DENOISPLIT_WEIGHT = 0.0  # dao3ch_config: usplit_w 1.0 -> pure muSplit

# Each condition: channels, checkpoint to predict, training phases in order.
# A later phase warm-starts from the previous phase's last checkpoint (A5).
CONDITIONS = {
    "4ch": dict(
        channels=[0, 1, 2, 3],
        predict_from="best",
        phases=[
            dict(
                suffix="",
                steps_per_epoch=2000,
                max_steps=800_000,
                early_stop_patience=200,
            )
        ],
    ),
    "ch01": dict(
        channels=[0, 1],
        predict_from="last",
        phases=[dict(suffix="", max_epochs=32)],  # an epoch = the whole dataset
    ),
    "ch23": dict(
        channels=[2, 3],
        predict_from="best",
        phases=[
            dict(
                suffix="",
                steps_per_epoch=2000,
                max_steps=380_000,
                early_stop_patience=200,
            ),
            dict(
                suffix="_lrdecay",
                steps_per_epoch=2000,
                max_steps=120_000,
                early_stop_patience=200,
                lr=5e-4,
                lr_scheduler="step",
                lr_step_epochs=30,
                lr_gamma=0.5,
            ),
        ],
    ),
}

# --- data (from scripts/microsplit_chicago_sch23.py) --------------------------
# Mirror: SIM1-49_merged.tif (T=49, C=4, 4096, 4096) and SIM201-263_merged.tif
# (T=63, ...) float32, TCYX through tifffile. disentangle concatenates them in
# this order (load-bearing: the split is positional) and splits with
# get_datasplit_tuples(0.1, 0.1). Input synthesized: 0.5/0.5 for a pair, the
# channel mean for all four (disentangle alpha 1/K).
DATA_ROOT = "/group/jug/public_html/microsplit/chicago_sch23_tiff"
FILES = ["SIM1-49_merged.tif", "SIM201-263_merged.tif"]
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1


def synthesize_input(target: np.ndarray) -> np.ndarray:
    if target.shape[1] == 2:
        return (0.5 * target[:, 0:1] + 0.5 * target[:, 1:2]).astype(target.dtype)
    return target.mean(axis=1, keepdims=True).astype(target.dtype)


def load_split_arrays(channels: list[int]):
    """(train, val, test) x (input, target) SCYX float32 arrays."""
    parts = []
    for f in FILES:
        data = tifffile.imread(os.path.join(DATA_ROOT, f))  # (T, C, Y, X)
        if data.ndim != 4:
            raise ValueError(f"{f}: expected TCYX, got {data.shape}")
        parts.append(np.take(data, channels, axis=1).astype(np.float32))
        del data
    target = np.concatenate(parts, axis=0)
    del parts
    tr, va, te = get_datasplit_tuples(VAL_FRACTION, TEST_FRACTION, len(target))
    inp = synthesize_input(target)
    print(
        f"[DATA] channels {channels}: {target.shape} -> train {len(tr)} val {len(va)} "
        f"test {len(te)} frames",
        flush=True,
    )
    return inp[tr], target[tr], inp[va], target[va], inp[te], target[te]


def main(args: argparse.Namespace) -> None:
    P.check_args(args)
    cond = CONDITIONS[args.condition]
    cond_dir = HERE / args.condition
    run = P.RunDirs(cond_dir, args.experiment_name)
    batch = args.batch_size or BATCH_SIZE
    n_out = len(cond["channels"])
    P.log_parameters(
        args,
        {
            "condition_dir": cond_dir,
            "run_dir": run.root,
            "PATCH_SIZE": PATCH_SIZE,
            "BATCH_SIZE": batch,
            **cond,
        },
    )
    tr_in, tr_tg, va_in, va_tg, te_in, te_tg = load_split_arrays(cond["channels"])
    if not (args.skip_n2v and args.skip_nm):
        print("[NM] pure muSplit: no N2V, no noise models", flush=True)

    def config_for(phase: dict):
        return P.microsplit_config(
            run.experiment + phase["suffix"],
            axes=AXES,
            patch_size=PATCH_SIZE,
            output_channels=n_out,
            multiscale_count=MULTISCALE_COUNT,
            batch_size=batch,
            num_workers=args.num_workers,
            seed=args.seed,
            denoisplit_weight=DENOISPLIT_WEIGHT,
            max_epochs=args.num_epochs or phase.get("max_epochs", 400),
            max_steps=args.max_steps or phase.get("max_steps"),
            steps_per_epoch=phase.get("steps_per_epoch"),
            early_stop_patience=phase.get("early_stop_patience"),
            lr=phase.get("lr"),
            lr_scheduler=phase.get("lr_scheduler", "plateau"),
            lr_step_epochs=phase.get("lr_step_epochs", 30),
            lr_gamma=phase.get("lr_gamma", 0.5),
            devices=args.devices,
            wandb=args.wandb,
            smoke=args.smoke,
        )

    # 3. MicroSplit, phase by phase
    if args.skip_train:
        careamist = P.make_careamist(config_for(cond["phases"][-1]), run.train)
        ckpt = Path(args.ckpt) if args.ckpt else P.single_file(cond_dir / "ckpts")
        if not args.skip_predict:
            P.compute_normalization(careamist, [tr_in], [tr_tg], [va_in], [va_tg])
    else:
        previous_last = None
        for phase in cond["phases"]:
            careamist = P.make_careamist(config_for(phase), run.train)
            P.attach_noise_model(careamist, None)
            sampled = phase.get("steps_per_epoch") and args.devices == 1
            dm = P.make_datamodule(
                careamist,
                [tr_in],
                [tr_tg],
                [va_in],
                [va_tg],
                samples_per_epoch=(
                    phase["steps_per_epoch"] * batch if sampled else None
                ),
            )
            best = P.train_stage(
                careamist,
                dm,
                work_dir=run.train,
                resume=not args.no_resume,
                init_ckpt=previous_last,
            )
            name = careamist.config.get_safe_experiment_name()
            previous_last = P.latest_last_checkpoint(run.train, name)
        ckpt = best if cond["predict_from"] == "best" else previous_last

    # 4. predict + metrics
    if args.skip_predict:
        return
    ev_in, ev_tg = (te_in, te_tg) if args.eval_split == "test" else (va_in, va_tg)
    ev_in, ev_tg = [ev_in], [ev_tg]
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
    P.add_common_args(parser, conditions=list(CONDITIONS), default_condition="ch23")
    main(parser.parse_args())
