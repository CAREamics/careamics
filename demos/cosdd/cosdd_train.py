"""Train COSDD through CAREamist with the settings of the COSDD paper.

Paper settings (Salmon & Krull, WACV 2025, supplementary 2.1 and 3):
"large" model = 14 Ladder VAE levels with 64 latent channels
(s_code_channels=128), 3 Gaussians, AR receptive field along x, random
256 x 256 crops, batch 4 x 4 accumulated batches, Adamax lr 2e-3,
ReduceLROnPlateau (factor 0.1, patience 50), early stopping after 100 epochs
without improvement. 16 crops per 1024 x 1024 frame and epoch, as in the
reference, is what careamics' random patching gives by default.

Example:
    python cosdd_train.py --dataset conva --experiment-name cosdd_conva_large
    python cosdd_train.py --dataset conva --experiment-name smoke --smoke
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from cosdd_data import load, train_val_split

from careamics import CAREamist
from careamics.config.factories import create_advanced_cosdd_config

RUNS_DIR = Path("/group/jug/Igor/cosdd/runs")

MODELS = {
    "large": {"n_layers": 14, "s_code_channels": 128},
    "small": {"n_layers": 6, "s_code_channels": 64},
    "tiny": {"n_layers": 2, "s_code_channels": 16},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["conva", "actin"])
    p.add_argument("--experiment-name", required=True)
    p.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    p.add_argument(
        "--model",
        default=None,
        choices=list(MODELS),
        help="default large, or tiny with --smoke",
    )
    p.add_argument("--n-gaussians", type=int, default=3)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--n-grad-batches", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--lr-patience", type=int, default=50)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--num-epochs", type=int, default=1000)
    p.add_argument("--num-steps", type=int, default=None, help="batches per epoch")
    p.add_argument("--max-time", default="00:24:00:00", help="DD:HH:MM:SS")
    p.add_argument("--precision", default="bf16-mixed")
    p.add_argument("--checkpointed", action="store_true")
    p.add_argument("--no-direct-denoiser", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="8 frames, 2 epochs of 5 batches, tiny model unless --model is given",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.model is None:
        args.model = "tiny" if args.smoke else "large"
    if args.smoke:
        args.num_epochs = 2
        args.num_steps = 5
        args.max_time = None

    data = load(args.dataset)
    frames = data.train[:8] if args.smoke else data.train
    train_idx, val_idx = train_val_split(len(frames), seed=args.seed)
    print(
        f"[DATA] {args.dataset}: {frames.shape} {frames.dtype}, "
        f"train {len(train_idx)} / val {len(val_idx)} frames, "
        f"noise direction {data.noise_direction}"
    )

    run_dir = args.runs_dir / args.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    trainer_params = {"precision": args.precision}
    if args.max_time is not None:
        trainer_params["max_time"] = args.max_time

    config = create_advanced_cosdd_config(
        experiment_name=args.experiment_name,
        data_type="array",
        axes="SYX",
        patch_size=(args.patch_size, args.patch_size),
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        num_steps=args.num_steps,
        noise_direction=data.noise_direction,
        use_direct_denoiser=not args.no_direct_denoiser,
        model_params={
            **MODELS[args.model],
            "n_gaussians": args.n_gaussians,
            "checkpointed": args.checkpointed,
        },
        n_grad_batches=args.n_grad_batches,
        num_workers=args.num_workers,
        trainer_params=trainer_params,
        optimizer="Adamax",
        optimizer_params={"lr": args.lr},
        lr_scheduler="ReduceLROnPlateau",
        lr_scheduler_params={"factor": 0.1, "patience": args.lr_patience},
        early_stopping_params={
            "monitor": "val_loss",
            "mode": "min",
            "patience": args.patience,
        },
        seed=args.seed,
    )
    print(config)

    careamist = CAREamist(config=config, work_dir=run_dir)
    n_params = sum(p.numel() for p in careamist.model.parameters())
    print(f"[MODEL] {args.model}: {n_params / 1e6:.2f} M parameters")

    t0 = time.time()
    careamist.train(train_data=frames[train_idx], val_data=frames[val_idx])
    train_min = (time.time() - t0) / 60

    ckpt_cb = careamist.trainer.checkpoint_callback
    info = {
        "dataset": args.dataset,
        "experiment_name": args.experiment_name,
        "model": args.model,
        "args": {k: str(v) for k, v in vars(args).items()},
        "train_frames": train_idx.tolist(),
        "val_frames": val_idx.tolist(),
        "epochs_done": careamist.trainer.current_epoch,
        "global_step": careamist.trainer.global_step,
        "train_minutes": round(train_min, 1),
        "best_model_path": str(ckpt_cb.best_model_path),
        "best_model_score": (
            float(ckpt_cb.best_model_score)
            if ckpt_cb.best_model_score is not None
            else None
        ),
        "last_model_path": str(ckpt_cb.last_model_path),
        "checkpoints": [str(p) for p in careamist.get_checkpoints()],
    }
    losses = careamist.get_losses()
    info["final_val_loss"] = float(np.asarray(losses.val_loss.value)[-1])
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2))

    print(
        f"[TRAIN] done: {info['epochs_done']} epochs, {info['global_step']} steps, "
        f"{info['train_minutes']} min, final val_loss {info['final_val_loss']:.4f}"
    )
    print(f"[CKPT] best {info['best_model_path']} ({info['best_model_score']})")
    print(f"[CKPT] last {info['last_model_path']}")
    print(f"[INFO] {run_dir / 'run_info.json'}")


if __name__ == "__main__":
    main()
