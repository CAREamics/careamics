"""Chicago-Sch23.

    python run_chicago_sch23.py --condition ch23                       # train + predict
    python run_chicago_sch23.py --condition 4ch --skip-train           # predict 4ch/ckpts
    python run_chicago_sch23.py --condition ch01 --smoke --experiment-name smoke

Outputs: /project/careamics/experiments/paper/msplit/chicago_sch23/<condition>/runs/<experiment-name>/{train,predict}
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

HERE = P.PAPER_ROOT / "chicago_sch23"

AXES = "SCYX"
PATCH_SIZE = (64, 64)
MULTISCALE_COUNT = 3
BATCH_SIZE = 32
DENOISPLIT_WEIGHT = 0.0

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
        phases=[dict(suffix="", max_epochs=32)],
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
        data = tifffile.imread(os.path.join(DATA_ROOT, f))
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
