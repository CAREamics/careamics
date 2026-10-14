"""Shared helpers for the run_<dataset>.py scripts in this folder.

Outputs: /project/careamics/experiments/paper/msplit/<dataset>/<condition>/runs/<experiment-name>/{n2v,noise_models,train,predict}
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional, Sequence

import lightning.pytorch as L
import numpy as np
import tifffile
import torch
from torch.utils.data import RandomSampler

from careamics import CAREamist
from careamics.config import create_advanced_microsplit_config, create_n2v_config
from careamics.lightning.data.data_module import CareamicsDataModule
from careamics.lvae_training.metrics import (
    _get_list_of_images_from_gt_pred,
    avg_psnr,
    avg_range_inv_psnr,
    compute_stats,
)
from careamics.models.constraints.model_constraints_factory import (
    get_model_constraints,
)
from careamics.noise_model import NoiseModelTrainer

PAPER_ROOT = Path("/project/careamics/experiments/paper/msplit")

Z_DIMS = [128] * 4
N_FILTERS = 64
GRID_SIZE = 32
PRECISION = 16
GRADIENT_CLIP_VAL = 0.5

SMOKE_N2V_STEPS = 2
SMOKE_NM_EPOCHS = 5
SMOKE_TRAIN_STEPS = 2
SMOKE_VAL_STEPS = 2
SMOKE_MMSE = 2


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    conditions: Sequence[str],
    default_condition: str,
) -> None:
    """Flags shared by every dataset script."""
    parser.add_argument(
        "--condition", choices=list(conditions), default=default_condition
    )
    parser.add_argument(
        "--experiment-name",
        default="e2e",
        help="Outputs go to <condition folder>/runs/<experiment-name>/.",
    )
    parser.add_argument(
        "--skip-n2v",
        action="store_true",
        help="Use the condition's n2v_ckpt/ (or --n2v-ckpt) for stage 2.",
    )
    parser.add_argument(
        "--skip-nm",
        action="store_true",
        help="Use the condition's noise_models/ (or --nm-paths).",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Use the condition's ckpts/ (or --ckpt) for prediction.",
    )
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument(
        "--n2v-ckpt", default=None, help="Overrides the bundled N2V ckpt."
    )
    parser.add_argument(
        "--nm-paths",
        nargs="+",
        default=None,
        help="Overrides the bundled noise models (one .npz per target).",
    )
    parser.add_argument(
        "--ckpt", default=None, help="Overrides the bundled MicroSplit ckpt."
    )
    parser.add_argument("--n2v-epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help=">1 = DDP training; stages 1, 2 and 4 then must be skipped.",
    )
    parser.add_argument(
        "--mmse", type=int, default=50, help="Posterior samples at predict."
    )
    parser.add_argument("--eval-split", choices=["test", "val"], default="test")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start training fresh even if this experiment has a last ckpt.",
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny budgets and crops: checks that every stage starts.",
    )


def check_args(args: argparse.Namespace) -> None:
    if args.devices > 1:
        bad = [
            s
            for s, run in (
                ("n2v", not args.skip_n2v),
                ("nm", not args.skip_nm),
                ("predict", not args.skip_predict),
            )
            if run
        ]
        if bad and not args.skip_train:
            raise SystemExit(
                f"--devices {args.devices}: DDP re-launches this script per rank, so "
                f"stages {bad} would run on every rank. Run them in a 1-GPU job and "
                "pass --skip-n2v --skip-nm --skip-predict here."
            )


class RunDirs:
    """<condition folder>/runs/<experiment>/{n2v,noise_models,train,predict}."""

    def __init__(self, condition_dir: Path, experiment: str) -> None:
        self.root = condition_dir / "runs" / experiment
        self.n2v = self.root / "n2v"
        self.nm = self.root / "noise_models"
        self.train = self.root / "train"
        self.predict = self.root / "predict"
        self.experiment = experiment
        for d in (self.n2v, self.nm, self.train, self.predict):
            d.mkdir(parents=True, exist_ok=True)


def log_parameters(args: argparse.Namespace, extra: dict) -> None:
    bar = "=" * 70
    print(f"\n{bar}\nRESOLVED PARAMETERS\n{bar}")
    for k in sorted(vars(args)):
        print(f"  {k:28s} = {getattr(args, k)!r}")
    print("-- dataset --")
    for k, v in extra.items():
        print(f"  {k:28s} = {v!r}")
    print(f"{bar}\n", flush=True)


def single_file(folder: Path, pattern: str = "*.ckpt") -> Path:
    hits = sorted(folder.glob(pattern))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one {pattern} in {folder}, got {hits}")
    return hits[0]


def bundled_noise_models(condition_dir: Path, n: int) -> list[Path]:
    paths = [
        condition_dir / "noise_models" / f"noise_model_ch{c}.npz" for c in range(n)
    ]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"bundled noise models missing: {missing}")
    return paths


def smoke_crop(
    stacks: Sequence[np.ndarray],
    *,
    yx: int = 256,
    z: int | None = None,
    n_samples: int = 1,
    n_stacks: int = 1,
) -> list[np.ndarray]:
    """First `n_stacks` stacks, first `n_samples` samples, Z[:z], centre YX crop.

    Works on any layout whose spatial axes are (..., [Z], Y, X) and whose first
    axis is S; `z` applies to the third-from-last axis.
    """
    out = []
    for s in list(stacks)[:n_stacks]:
        a = np.asarray(s)[:n_samples]
        y0 = max((a.shape[-2] - yx) // 2, 0)
        x0 = max((a.shape[-1] - yx) // 2, 0)
        a = a[..., y0 : y0 + yx, x0 : x0 + yx]
        if z is not None:
            a = a[..., :z, :, :]
        out.append(np.ascontiguousarray(a))
    return out


def smoke_crop_channel_last(stacks: Sequence[np.ndarray], **kw) -> list[np.ndarray]:
    """`smoke_crop` for (S, [Z], Y, X, C) stacks."""
    moved = [np.moveaxis(np.asarray(s), -1, 1) for s in stacks]
    return [
        np.ascontiguousarray(np.moveaxis(a, 1, -1)) for a in smoke_crop(moved, **kw)
    ]


_VAL_LOSS = re.compile(r"_(-?[0-9]+\.[0-9]+)\.ckpt$")


def best_checkpoint(ckpts: Sequence[Path]) -> Optional[Path]:
    """Lowest val_loss among careamics checkpoint names `<exp>_<ep>_step_<s>_<loss>.ckpt`."""
    best, best_loss = None, float("inf")
    for ck in ckpts:
        if ck.name.endswith("_last.ckpt"):
            continue
        m = _VAL_LOSS.search(ck.name)
        if m and float(m.group(1)) < best_loss:
            best, best_loss = Path(ck), float(m.group(1))
    return best


def version_dirs(work_dir: Path, experiment: str) -> list[Path]:
    """CAREamist checkpoint folders `<work_dir>/checkpoints/<exp>_<version>`, oldest first."""
    root = work_dir / "checkpoints"
    dirs = [
        d
        for d in root.glob(f"{experiment}_*")
        if d.is_dir() and d.name[len(experiment) + 1 :].isdigit()
    ]
    return sorted(dirs, key=lambda d: int(d.name[len(experiment) + 1 :]))


def latest_last_checkpoint(work_dir: Path, experiment: str) -> Optional[Path]:
    for d in reversed(version_dirs(work_dir, experiment)):
        last = d / f"{experiment}_last.ckpt"
        if last.exists():
            return last
    return None


def best_training_checkpoint(work_dir: Path, experiment: str) -> Optional[Path]:
    ckpts = [c for d in version_dirs(work_dir, experiment) for c in d.glob("*.ckpt")]
    return best_checkpoint(ckpts)


def load_weights(model: torch.nn.Module, ckpt_path: Path | str) -> None:
    """Load a MicroSplit checkpoint's weights into `model` (A5).

    Accepts the noise-model layouts our checkpoints were saved with: the noise
    model as `noise_model.*` state_dict entries (2026-09-25..28) or under its own
    top-level key (later). Neither is a weight of the LVAE. `model.parameter_net.*`
    aliases `model.output_layer.*`, so it may be reported missing (bug catalog B2).
    Anything else that does not match raises.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    state = {k: v for k, v in state.items() if not k.startswith("noise_model.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [
        k for k in missing if not k.startswith(("noise_model.", "model.parameter_net."))
    ]
    unexpected = [k for k in unexpected if not k.startswith("noise_model.")]
    if missing or unexpected:
        raise RuntimeError(
            f"{ckpt_path}: {len(missing)} missing / {len(unexpected)} unexpected keys, "
            f"e.g. {missing[:3]} {unexpected[:3]}"
        )
    print(
        f"[WEIGHTS] loaded {len(state)} tensors from {ckpt_path} "
        f"(epoch {ckpt.get('epoch')}, step {ckpt.get('global_step')})",
        flush=True,
    )


def predictable_checkpoint(
    ckpt_path: Path | str, model: torch.nn.Module, out_dir: Path
) -> Path:
    """A checkpoint careamics' predict can load strictly into `model` (A5).

    Lightning restores `ckpt_path` with strict key matching. Checkpoints saved
    with the 2026-09-25..28 noise-model API carry `noise_model.*` state_dict
    entries, and pre-alias ones lack `model.parameter_net.*`; both are rewritten
    into a copy in `out_dir`. The original is never touched; any other key
    mismatch raises.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    expected = model.state_dict()
    fixed = {k: v for k, v in state.items() if not k.startswith("noise_model.")}
    for k in expected:
        if k.startswith("model.parameter_net.") and k not in fixed:
            alias = k.replace("model.parameter_net.", "model.output_layer.", 1)
            if alias in fixed:
                fixed[k] = fixed[alias]
    expected_keys = {k for k in expected if not k.startswith("noise_model.")}
    if set(fixed) != expected_keys:
        diff = sorted(set(fixed) ^ expected_keys)
        raise RuntimeError(
            f"{ckpt_path}: state_dict does not match the model: {diff[:10]}"
        )
    if len(fixed) == len(state) and set(fixed) == set(state):
        return Path(ckpt_path)
    ckpt["state_dict"] = fixed
    out = out_dir / f"{Path(ckpt_path).stem}_clean.ckpt"
    torch.save(ckpt, out)
    print(
        f"[CKPT] {ckpt_path} -> {out} (dropped {len(state) - len(set(state) & set(fixed))} "
        f"noise_model keys, added {len(set(fixed) - set(state))} alias keys)",
        flush=True,
    )
    return out


def n2v_stage(
    nm_input: Sequence[np.ndarray],
    *,
    work_dir: Path,
    name: str,
    axes: str,
    n_channels: int,
    patch_size: Sequence[int],
    epochs: int,
    batch_size: int,
    seed: int,
    tile_size: Sequence[int],
    tile_overlap: Sequence[int],
    pick: str = "best",
    n2v_ckpt: Path | str | None = None,
    smoke: bool = False,
) -> list[np.ndarray]:
    """Train N2V (or load `n2v_ckpt`) and return its prediction of every input stack.

    `pick` chooses the checkpoint that predicts after training: "best" = lowest
    val_loss among the saved ones (A6), "last" = the final one. The input layout
    (channel last) is kept.
    """
    if n2v_ckpt is not None:
        print(f"[N2V] predicting from {n2v_ckpt}; no N2V training", flush=True)
        careamist = CAREamist(
            checkpoint_path=str(n2v_ckpt),
            work_dir=str(work_dir),
            enable_progress_bar=False,
        )
        checkpoint = str(n2v_ckpt)
    else:
        config = create_n2v_config(
            experiment_name=f"{name}_n2v",
            data_type="array",
            axes=axes,
            n_channels=n_channels,
            patch_size=tuple(patch_size),
            batch_size=batch_size,
            num_epochs=1 if smoke else epochs,
            num_steps=SMOKE_N2V_STEPS if smoke else None,
        )
        config.data_config.seed = seed
        careamist = CAREamist(
            config=config, work_dir=str(work_dir), enable_progress_bar=False
        )
        careamist.train(train_data=list(nm_input))
        checkpoint = None
        if pick == "best":
            best = best_checkpoint(careamist.get_checkpoints())
            if best is not None:
                checkpoint = str(best)
        print(
            f"[N2V] predicting from {checkpoint or 'the last checkpoint'}", flush=True
        )
    prediction, _ = careamist.predict(
        list(nm_input),
        tile_size=tuple(tile_size),
        tile_overlap=tuple(tile_overlap),
        checkpoint=checkpoint,
    )
    prediction = [np.asarray(p, dtype=np.float32) for p in prediction]
    for p, x in zip(prediction, nm_input):
        if p.shape != np.shape(x):
            raise RuntimeError(
                f"N2V prediction {p.shape} does not match its input {np.shape(x)}"
            )
    if len(prediction) != len(nm_input):
        raise RuntimeError(f"N2V returned {len(prediction)} stacks for {len(nm_input)}")
    return prediction


def pixel_pairs(stacks: Sequence[np.ndarray]) -> np.ndarray:
    """Channel-last stacks of any shapes -> one (1, C, 1, N) SCYX pixel array (A2)."""
    n_ch = np.shape(stacks[0])[-1]
    flat = np.concatenate([np.asarray(s).reshape(-1, n_ch) for s in stacks], axis=0)
    return np.ascontiguousarray(flat.T[None, :, None, :])


def fit_noise_models(
    signal: Sequence[np.ndarray],
    observation: Sequence[np.ndarray],
    out_dir: Path,
    *,
    n_gaussian: int,
    n_coeff: int,
    min_sigma: float,
    n_epochs: int,
    domain: str,
) -> list[Path]:
    """One GMM per channel from channel-last (signal, observation) stacks.

    domain = "signal": careamics' `train_from_pairs`, GMM range = signal range.
    domain = "union":  GMM range = union of signal and observation ranges (A1),
                       the recipe of the published disentangle noise models.
    The signal is never clipped. Non-finite weights abort before saving (A3).
    """
    if domain not in ("signal", "union"):
        raise ValueError(f"domain must be 'signal' or 'union', got {domain!r}")
    sig, obs = pixel_pairs(signal), pixel_pairs(observation)
    trainer = NoiseModelTrainer(
        n_gaussian=n_gaussian, n_coeff=n_coeff, min_sigma=min_sigma
    )
    ranges = []
    if domain == "signal":
        trainer.train_from_pairs(
            signal=sig, observation=obs, axes="SCYX", n_epochs=n_epochs
        )
        ranges = [
            (float(nm.min_signal.item()), float(nm.max_signal.item()))
            for nm in trainer.noise_models
        ]
    else:
        trainer.noise_models, trainer.histograms = [], []
        trainer.channel_indices, trainer.train_losses = [], []
        for c in range(sig.shape[1]):
            lo = float(min(sig[:, c].min(), obs[:, c].min()))
            hi = float(max(sig[:, c].max(), obs[:, c].max()))
            ranges.append((lo, hi))
            nm, losses = trainer._train_single_channel(
                signal=sig[:, c],
                observation=obs[:, c],
                n_epochs=n_epochs,
                learning_rate=1e-1,
                batch_size=250000,
                min_signal=lo,
                max_signal=hi,
            )
            trainer.noise_models.append(nm)
            trainer.channel_indices.append(c)
            trainer.train_losses.append(losses)
    for c, nm in enumerate(trainer.noise_models):
        if not torch.isfinite(nm.weight.detach()).all():
            raise RuntimeError(
                f"ch{c}: non-finite GMM weights after the fit; not saving."
            )
    paths = trainer.save(out_dir)
    params = dict(
        n_gaussian=n_gaussian,
        n_coeff=n_coeff,
        min_sigma=min_sigma,
        n_epochs=n_epochs,
        domain=domain,
        ranges=ranges,
    )
    (out_dir / "nm_params.json").write_text(json.dumps(params, indent=2))
    print(f"[NM] saved {[str(p) for p in paths]}  {params}", flush=True)
    return paths


def microsplit_config(
    name: str,
    *,
    axes: str,
    patch_size: Sequence[int],
    output_channels: int,
    multiscale_count: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    denoisplit_weight: float,
    max_epochs: int,
    max_steps: Optional[int] = None,
    steps_per_epoch: Optional[int] = None,
    early_stop_patience: Optional[int] = None,
    lr: Optional[float] = None,
    lr_scheduler: str = "plateau",
    lr_step_epochs: int = 30,
    lr_gamma: float = 0.5,
    encoder_conv_strides: Optional[Sequence[int]] = None,
    decoder_conv_strides: Optional[Sequence[int]] = None,
    augmentations: Optional[Sequence[str]] = None,
    devices: int = 1,
    wandb: bool = False,
    smoke: bool = False,
) -> Any:
    """The MicroSplit configuration of one bundled run, through careamics' factory.

    `denoisplit_weight` 0 is pure muSplit (no noise model). Early stopping is
    always explicit: careamics' MicroSplit default is EarlyStopping with
    Lightning's patience 3.
    """
    trainer_params: dict[str, Any] = {
        "max_epochs": max_epochs,
        "precision": PRECISION,
        "gradient_clip_algorithm": "value",
        "gradient_clip_val": GRADIENT_CLIP_VAL,
    }
    if max_steps is not None:
        trainer_params.update(max_steps=max_steps, max_epochs=-1)
    if steps_per_epoch is not None:
        trainer_params["limit_train_batches"] = steps_per_epoch
    if devices > 1:
        trainer_params.update(
            accelerator="gpu",
            devices=devices,
            strategy="ddp_find_unused_parameters_true",
            sync_batchnorm=True,
        )
    if smoke:
        trainer_params.update(
            max_epochs=1,
            max_steps=-1,
            limit_train_batches=SMOKE_TRAIN_STEPS,
            limit_val_batches=SMOKE_VAL_STEPS,
        )
    kwargs: dict[str, Any] = dict(
        experiment_name=name,
        data_type="array",
        axes=axes,
        patch_size=tuple(patch_size),
        batch_size=batch_size,
        output_channels=output_channels,
        num_epochs=trainer_params["max_epochs"],
        multiscale_count=multiscale_count,
        model_params={"z_dims": Z_DIMS, "n_filters": N_FILTERS},
        gaussian_likelihood_weight=1.0 - denoisplit_weight,
        noise_model_likelihood_weight=denoisplit_weight,
        num_workers=num_workers,
        trainer_params=trainer_params,
        early_stopping_params=(
            {}
            if early_stop_patience is None
            else {
                "monitor": "val_loss",
                "mode": "min",
                "patience": early_stop_patience,
                "check_on_train_epoch_end": False,
            }
        ),
        logger="wandb" if wandb else "none",
        seed=seed,
    )
    if augmentations is not None:
        kwargs["augmentations"] = list(augmentations)
    if lr is not None:
        kwargs["optimizer_params"] = {"lr": lr, "weight_decay": 0}
    if lr_scheduler == "step":
        kwargs["lr_scheduler"] = "StepLR"
        kwargs["lr_scheduler_params"] = {"step_size": lr_step_epochs, "gamma": lr_gamma}
    if encoder_conv_strides is not None:
        kwargs["encoder_conv_strides"] = list(encoder_conv_strides)
    if decoder_conv_strides is not None:
        kwargs["decoder_conv_strides"] = list(decoder_conv_strides)
    return create_advanced_microsplit_config(**kwargs)


class SampledDataModule(CareamicsDataModule):
    """`CareamicsDataModule` whose training epoch is `samples_per_epoch` draws (A4).

    disentangle's epochs are `limit_train_batches` = 2000 steps, and its LR
    plateau patience and early stopping count those epochs. A dataset with fewer
    than 2000 x batch patches cannot fill that epoch through `limit_train_batches`,
    so the sampler draws back-to-back permutations: every patch is seen before any
    repeats. Single GPU only.
    """

    def __init__(self, *args: Any, samples_per_epoch: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.samples_per_epoch = samples_per_epoch

    def _sampler(self, stage: str):
        if stage == "train":
            return RandomSampler(
                self.train_dataset,
                replacement=False,
                num_samples=self.samples_per_epoch,
            )
        return super()._sampler(stage)


def make_careamist(config: Any, work_dir: Path) -> CAREamist:
    return CAREamist(config=config, work_dir=str(work_dir), enable_progress_bar=False)


def make_datamodule(
    careamist: CAREamist,
    train_in,
    train_tg,
    val_in,
    val_tg,
    samples_per_epoch: Optional[int] = None,
) -> CareamicsDataModule:
    """careamics' datamodule on the CAREamist's own data config (stats land there)."""
    kwargs = dict(
        train_data=list(train_in),
        train_data_target=list(train_tg),
        val_data=list(val_in),
        val_data_target=list(val_tg),
        model_constraints=get_model_constraints(
            careamist.config.algorithm_config.model
        ),
    )
    if samples_per_epoch:
        return SampledDataModule(
            careamist.config.data_config, samples_per_epoch=samples_per_epoch, **kwargs
        )
    return CareamicsDataModule(careamist.config.data_config, **kwargs)


def attach_noise_model(
    careamist: CAREamist, nm_paths: Optional[Sequence[Path]]
) -> None:
    if (
        nm_paths
        and careamist.config.algorithm_config.loss.noise_model_likelihood_weight > 0
    ):
        careamist.model.set_noise_model([str(p) for p in nm_paths])
        print(f"[NM] training with {[str(p) for p in nm_paths]}", flush=True)
    else:
        print("[NM] no noise model (pure muSplit)", flush=True)


def train_stage(
    careamist: CAREamist,
    dm: CareamicsDataModule,
    *,
    work_dir: Path,
    resume: bool = True,
    init_ckpt: Path | str | None = None,
) -> Optional[Path]:
    """Fit, resuming this experiment's newest last ckpt; returns the best ckpt.

    `init_ckpt` warm-starts the weights on a fresh run (A5); a resume ignores it
    because the resumed checkpoint already holds the weights.
    """
    name = careamist.config.get_safe_experiment_name()
    last = latest_last_checkpoint(work_dir, name) if resume else None
    if last is not None:
        print(f"[TRAIN] resuming {last}", flush=True)
    elif init_ckpt is not None:
        print(f"[TRAIN] warm start from {init_ckpt}", flush=True)
        load_weights(careamist.model, init_ckpt)
    L.seed_everything(careamist.config.data_config.seed, workers=True)
    careamist.trainer.fit(
        careamist.model, datamodule=dm, ckpt_path=str(last) if last else None
    )
    best = best_training_checkpoint(work_dir, name) or latest_last_checkpoint(
        work_dir, name
    )
    print(f"[TRAIN] done; best checkpoint {best}", flush=True)
    return best


def predict_stage(
    careamist: CAREamist,
    eval_input: Sequence[np.ndarray],
    *,
    ckpt: Path | str,
    out_dir: Path,
    patch_size: Sequence[int],
    grid_z: Optional[int] = None,
    mmse: int,
    batch_size: Optional[int] = None,
    seed: int = 42,
) -> list[np.ndarray]:
    """Tiled MMSE prediction of each eval stack from `ckpt`, returned channel last.

    Tiles overlap by patch - 32 in Y/X. In Z the tiles abut unless `grid_z`
    is given (disentangle's (grid_z, 32, 32) eval grid: overlap = depth - grid_z).
    """
    overlap = [max(p - GRID_SIZE, 0) for p in patch_size]
    if grid_z is not None:
        overlap[0] = patch_size[0] - grid_z
    ckpt = predictable_checkpoint(ckpt, careamist.model, out_dir)
    print(
        f"[PREDICT] {ckpt}: tile {tuple(patch_size)} overlap {tuple(overlap)} "
        f"mmse {mmse}",
        flush=True,
    )
    L.seed_everything(seed, workers=True)
    out = careamist.predict(
        list(eval_input),
        tile_size=tuple(patch_size),
        tile_overlap=tuple(overlap),
        batch_size=batch_size,
        n_predictions=mmse if mmse > 1 else None,
        checkpoint=str(ckpt),
    )
    preds = out[0]
    if len(preds) != len(eval_input):
        raise RuntimeError(
            f"{len(preds)} predictions for {len(eval_input)} eval stacks"
        )
    return [np.moveaxis(np.asarray(p), 1, -1) for p in preds]


def evaluate(
    preds: Sequence[np.ndarray],
    gts: Sequence[np.ndarray],
    out_dir: Path,
    *,
    tag: str,
    save: bool = True,
) -> dict:
    """compute_stats (range-invariant PSNR, MicroSSIM, MS3IM, ...) plus plain PSNR.

    Inputs are channel-last stacks. Prints the `Metrics:` line the logs are read by.
    """
    metrics = compute_stats(list(gts), list(preds))
    plain, rinv = [], []
    for ch in range(np.shape(gts[0])[-1]):
        g, p = _get_list_of_images_from_gt_pred(list(gts), list(preds), ch)
        plain.append(avg_psnr(g, p))
        rinv.append(avg_range_inv_psnr(g, p))
    metrics["psnr"] = plain
    print(f"PSNR (plain, per channel)          : {plain}")
    print(f"PSNR (range-invariant, per channel): {rinv}")
    print(f"Metrics: {metrics}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"metrics_{tag}.json").write_text(
        json.dumps(metrics, indent=2, default=float)
    )
    if save:
        for i, p in enumerate(preds):
            tifffile.imwrite(
                out_dir / f"pred_{tag}_{i:03d}.tif", np.asarray(p, np.float32)
            )
    return metrics


def compute_normalization(
    careamist: CAREamist, train_in, train_tg, val_in, val_tg
) -> None:
    """Resolve the training normalization into `careamist.config` without training.

    The statistics are computed from the training data exactly as a training run
    does (careamics resolves them when the training dataset is built), which is
    what a predict-only run needs: none of the bundled checkpoints except LIF24
    5 ms stores them.
    """
    dm = make_datamodule(careamist, train_in, train_tg, val_in, val_tg)
    dm.setup("fit")
    print(f"[NORM] {careamist.config.data_config.normalization}", flush=True)
