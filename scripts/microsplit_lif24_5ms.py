"""
MicroSplit training on the LIF24 5ms 2-channel dataset (TIFF).

Channel layout:
  target 0 = raw ch 0  (Nucleus)
  target 1 = raw ch 1  (MicroTubules)
  input    = raw ch 8  (superimposed "01")

Examples
--------
Full training from scratch (fits N2V + noise models, then trains LVAE):

    python scripts/microsplit_lif24_5ms.py \\
        --experiment-name ht_lif24_5ms_ngds_v3 \\
        --num-epochs 40

Default with pretrained noise models:

    python scripts/microsplit_lif24_5ms.py \\
        --experiment-name ht_lif24_5ms_ngds_40ep_gpuq \\
        --noise-model-paths scripts/noise_models/5ms_ngds/noise_model_Ch{0,1}.npz

Predict-only from a checkpoint:

    python scripts/microsplit_lif24_5ms.py \\
        --experiment-name ht_lif24_5ms_ngds_40ep_gpuq \\
        --skip-training \\
        --pretrained-ckpt scripts/lvae_checkpoints/5ms/<EXP>/checkpoints/last.ckpt
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as L
import tifffile
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler

from careamics import CAREamist
from careamics.config import (
    MicroSplitDataConfig,
    create_advanced_microsplit_config,
    create_n2v_config,
)
from careamics.dataset.factory import (
    PairedInputTarget,
    create_microsplit_dataset,
    create_microsplit_pred_dataset,
)
from careamics.dataset.factory.factory import TrainValData
from careamics.lightning.modules.microsplit_module import MicroSplitModule
from careamics.lightning.prediction.convert_prediction import convert_prediction
from careamics.lvae_training.dataset.utils.data_utils import get_datasplit_tuples
from careamics.lvae_training.metrics import RangeInvariantPsnr, compute_stats
from careamics.noise_model import NoiseModelTrainer
from careamics.utils.get_device import get_device

# ---------------------------------------------------------------------------
# Fixed parameters
# ---------------------------------------------------------------------------

# path can be set as a CLI arg
DEFAULT_DATA_ROOT = Path("/group/jug/public_html/microsplit/ht_lif24_tiff")
EXPOSURE_DURATION = "5ms"
HIGHSNR_EXPOSURE_DURATION = "500ms"
CH_IDX_LIST = [0, 1, 8]  # targets [Nucleus, MicroTubules] + superimposed input "01"

BASE_DIR = Path(__file__).resolve().parent

# Fixed model / trainer knobs (never varied between experiments)
Z_DIMS = [128] * 4
N_FILTERS = 64  # post PR #1049: single field, applies to both encoder and decoder
MULTISCALE_COUNT = 3
OUTPUT_CHANNELS = len(CH_IDX_LIST) - 1  # 2
MMSE_COUNT = 1
GRID_SIZE = 32
TRAINER_PRECISION = 16
TRAINER_GRADIENT_CLIP_VAL = 0.5
TRAINER_GRADIENT_CLIP_ALGORITHM = "value"

# Noise-model fitting (used only when NM paths not supplied)
NM_N_GAUSSIAN = 3
NM_N_COEFF = 3
NM_MIN_SIGMA = 200.0
NM_N_EPOCHS = 2000

VAL_FRACTION = 0.1
TEST_FRACTION = 0.1

N_PREVIEW_SAMPLES = 4


# ---------------------------------------------------------------------------
# TIFF loading
# ---------------------------------------------------------------------------


def _datafiles(exposure: str) -> list[str]:
    return [f"Set{i}/uSplit_{exposure}.tif" for i in range(1, 7)]


def _load_one_fpath(fpath: str, channel_list: list[int]) -> np.ndarray:
    """Read a TIFF, select channels, return (P, Y, X, C) SYXC."""
    data = tifffile.imread(fpath)
    data = data[:, channel_list, ...]
    data = np.swapaxes(data[..., None], 1, 4)[:, 0]
    fname_prefix = "_".join(os.path.basename(fpath).split(".")[0].split("_")[:-1])
    if fname_prefix == "uSplit_20022025_001":
        data = np.delete(data, 2, axis=0)
    elif fname_prefix == "uSplit_14022025":
        data = np.delete(data, [17, 19], axis=0)
    return data


def _load_data(datadir: str, channel_list: list[int], exposure: str) -> np.ndarray:
    return np.concatenate(
        [
            _load_one_fpath(os.path.join(datadir, f), channel_list)
            for f in _datafiles(exposure)
        ],
        axis=0,
    )


def _syxc_to_scyx(arr: np.ndarray) -> np.ndarray:
    return np.moveaxis(arr, -1, 1)


def load_split_arrays(
    datadir: str, exposure: str, channel_list: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (train, val, test) x (input, target) SCYX arrays."""
    data = _load_data(datadir, channel_list=channel_list, exposure=exposure)
    train_idx, val_idx, test_idx = get_datasplit_tuples(
        VAL_FRACTION, TEST_FRACTION, len(data)
    )
    data = data.astype(np.float32)
    input_arr = _syxc_to_scyx(data[..., -1:])  # (N, 1, Y, X)
    target_arr = _syxc_to_scyx(data[..., :-1])  # (N, 2, Y, X)
    return (
        input_arr[train_idx],
        target_arr[train_idx],
        input_arr[val_idx],
        target_arr[val_idx],
        input_arr[test_idx],
        target_arr[test_idx],
    )


# ---------------------------------------------------------------------------
# LightningDataModule
# ---------------------------------------------------------------------------


class MicroSplitNgDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_config: MicroSplitDataConfig,
        val_config: MicroSplitDataConfig,
        train_input: np.ndarray,
        train_target: np.ndarray,
        val_input: np.ndarray,
        val_target: np.ndarray,
        pred_config: MicroSplitDataConfig | None = None,
        pred_input: np.ndarray | None = None,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.train_config = train_config
        self.val_config = val_config
        self.pred_config = pred_config
        self.train_input = [train_input]
        self.train_target = [train_target]
        self.val_input = [val_input]
        self.val_target = [val_target]
        self.pred_input = [pred_input] if pred_input is not None else None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._data = TrainValData(
            train_data=self.train_input,
            val_data=self.val_input,
            train_data_target=self.train_target,
            val_data_target=self.val_target,
        )
        self.train_dataset = None
        self.val_dataset = None
        self.predict_dataset = None

    def setup(self, stage: str) -> None:
        if stage in ("fit", "validate"):
            if self.train_dataset is None:
                self.train_dataset = create_microsplit_dataset(
                    config=self.train_config,
                    data=PairedInputTarget(
                        input_data=self.train_input,
                        target_data=self.train_target,
                    ),
                )
            if self.val_dataset is None:
                shared_norm = self.train_config.normalization
                val_cfg = self.val_config.model_copy(
                    update={"normalization": shared_norm}
                )
                self.val_dataset = create_microsplit_dataset(
                    config=val_cfg,
                    data=PairedInputTarget(
                        input_data=self.val_input,
                        target_data=self.val_target,
                    ),
                )
        elif stage == "predict":
            if self.pred_config is None or self.pred_input is None:
                raise ValueError("predict stage requires pred_config + pred_input")
            self.predict_dataset = create_microsplit_pred_dataset(
                config=self.pred_config,
                input_data=self.pred_input,
            )

    def _maybe_distributed_sampler(self, dataset, shuffle: bool):
        """Shard `dataset` across DDP ranks; return None outside DDP."""
        if self.trainer is None or self.trainer.world_size <= 1:
            return None
        return DistributedSampler(
            dataset,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        sampler = self._maybe_distributed_sampler(self.train_dataset, shuffle=True)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            shuffle=(sampler is None),
            collate_fn=default_collate,
        )

    def val_dataloader(self):
        sampler = self._maybe_distributed_sampler(self.val_dataset, shuffle=False)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            shuffle=False,
            collate_fn=default_collate,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=default_collate,
        )


# ---------------------------------------------------------------------------
# Noise model training
# ---------------------------------------------------------------------------


def prepare_noise_model_data(data_path: Path, exposure: str) -> np.ndarray:
    """Return the train-split subset in SYXC layout (float32)."""
    data = _load_data(str(data_path), CH_IDX_LIST, exposure).astype(np.float32)
    train_idx, _, _ = get_datasplit_tuples(VAL_FRACTION, TEST_FRACTION, len(data))
    return data[train_idx]


def train_n2v(
    nm_input: np.ndarray,
    experiment_name: str,
    noise_model_dir: Path,
    n2v_num_epochs: int,
    patch_size: tuple[int, int],
    batch_size: int,
    seed: int,
) -> np.ndarray:
    """Train N2V on the multi-channel training data and return its predictions.

    Returns
    -------
    (S, Y, X, C) SYXC float32 array of N2V predictions (same layout as `nm_input`).
    """
    config = create_n2v_config(
        experiment_name=f"{experiment_name}_n2v",
        data_type="array",
        axes="SYXC",
        n_channels=len(CH_IDX_LIST),
        patch_size=patch_size,
        batch_size=batch_size,
        num_epochs=n2v_num_epochs,
    )
    config.data_config.seed = seed
    careamist = CAREamist(config=config, work_dir=str(noise_model_dir))
    careamist.train(train_data=nm_input)
    prediction, _ = careamist.predict(nm_input, tile_size=(256, 256))
    return np.concatenate(prediction, axis=0)


def fit_noise_models(
    signal: np.ndarray,
    observation: np.ndarray,
    noise_model_dir: Path,
) -> list[Path]:
    """Fit one GMM per target channel via `NoiseModelTrainer.train_from_pairs`.

    Both `signal` (N2V predictions) and `observation` (raw noisy) are SYXC.
    Only target channels (all but last, which is the input) are used.
    """
    noise_model_dir.mkdir(parents=True, exist_ok=True)
    # drop the trailing input channel — fit NM only for target channels
    signal_targets = signal[..., :-1]  # (S, Y, X, C_out)
    observation_targets = observation[..., :-1]

    trainer = NoiseModelTrainer(
        n_gaussian=NM_N_GAUSSIAN,
        n_coeff=NM_N_COEFF,
        min_sigma=NM_MIN_SIGMA,
    )
    trainer.train_from_pairs(
        signal=signal_targets,
        observation=observation_targets,
        signal_axes="SYXC",
        observation_axes="SYXC",
        n_epochs=NM_N_EPOCHS,
    )
    return trainer.save(noise_model_dir)


# ---------------------------------------------------------------------------
# Config build
# ---------------------------------------------------------------------------


def build_microsplit_config(
    experiment_name: str,
    nm_paths: Optional[list[Path]],
    num_epochs: int,
    batch_size: int,
    num_workers: int,
    patch_size: tuple[int, int],
    use_wandb: bool,
    seed: int,
    mmse_count: int = MMSE_COUNT,
) -> Any:
    if nm_paths:
        nm_config = NoiseModelTrainer.config_from_paths(nm_paths)
        musplit_w, denoisplit_w = 0.1, 0.9  # denoiSplit-heavy (factory default)
    else:
        nm_config = None
        musplit_w, denoisplit_w = 1.0, 0.0  # pure µSplit

    config = create_advanced_microsplit_config(
        experiment_name=experiment_name,
        data_type="array",
        axes="SCYX",
        patch_size=patch_size,
        output_channels=OUTPUT_CHANNELS,
        multiscale_count=MULTISCALE_COUNT,
        batch_size=batch_size,
        num_epochs=num_epochs,
        mmse_count=mmse_count,
        noise_model=nm_config,
        musplit_weight=musplit_w,
        denoisplit_weight=denoisplit_w,
        model_params={
            "z_dims": Z_DIMS,
            "n_filters": N_FILTERS,
        },
        train_dataloader_params={"num_workers": num_workers},
        val_dataloader_params={"num_workers": num_workers},
        logger="wandb" if use_wandb else "none",
        trainer_params={
            "max_epochs": num_epochs,
            "precision": TRAINER_PRECISION,
            "gradient_clip_algorithm": TRAINER_GRADIENT_CLIP_ALGORITHM,
            "gradient_clip_val": TRAINER_GRADIENT_CLIP_VAL,
        },
        seed=seed,
    )
    return config


# ---------------------------------------------------------------------------
# Train / predict / eval
# ---------------------------------------------------------------------------


def load_pretrained_model(model: MicroSplitModule, ckpt_path: str) -> None:
    device = get_device()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    unexpected = [k for k in unexpected if not k.startswith("noise_model.")]
    missing = [k for k in missing if not k.startswith("noise_model.")]
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint {ckpt_path!r} does not match MicroSplitModule "
            f"({len(missing)} missing, {len(unexpected)} unexpected keys)."
        )


def create_trainer(
    config: Any,
    output_dir: Path,
    experiment_name: str,
    *,
    devices: int = 1,
    strategy: str = "auto",
    sync_batchnorm: bool = False,
) -> Trainer:
    """Build a Lightning Trainer. `devices=1` preserves single-GPU behavior."""
    output_dir.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = dict(
        max_epochs=config.training_config.trainer_params["max_epochs"],
        precision=config.training_config.trainer_params["precision"],
        gradient_clip_algorithm=config.training_config.trainer_params[
            "gradient_clip_algorithm"
        ],
        gradient_clip_val=config.training_config.trainer_params["gradient_clip_val"],
        default_root_dir=output_dir,
        callbacks=[
            ModelCheckpoint(
                dirpath=output_dir / "checkpoints",
                filename=f"ht_lif24_{EXPOSURE_DURATION}_{experiment_name}",
                save_last=True,
            ),
            TQDMProgressBar(refresh_rate=50),
        ],
        logger=(
            WandbLogger(project="microsplit_lightning_api", name=experiment_name)
            if config.training_config.logger == "wandb"
            else None
        ),
    )
    if devices > 1:
        kwargs.update(
            accelerator="gpu",
            devices=devices,
            strategy=strategy,
            sync_batchnorm=sync_batchnorm,
            num_nodes=1,
            # Datamodule installs its own DistributedSampler; don't re-wrap.
            use_distributed_sampler=False,
        )
    return Trainer(**kwargs)


def _train_val_cfgs(config: Any) -> tuple[MicroSplitDataConfig, MicroSplitDataConfig]:
    train_cfg: MicroSplitDataConfig = config.data_config
    return train_cfg, train_cfg.convert_mode("validating")


def _predict_cfg(config: Any, patch_size: tuple[int, int]) -> MicroSplitDataConfig:
    train_cfg: MicroSplitDataConfig = config.data_config
    overlap = tuple(p - GRID_SIZE for p in patch_size)
    return train_cfg.convert_mode(
        "predicting",
        new_patch_size=patch_size,
        overlap_size=overlap,
    )


def _psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """Range-invariant PSNR (matches `compute_stats`) for one 2D target/pred pair."""
    gt = target.astype(np.float32)[None]  # (1, H, W)
    pr = pred.astype(np.float32)[None]
    return float(RangeInvariantPsnr(gt, pr).item())


def save_preview_images(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_dir: Path,
    n_samples: int = N_PREVIEW_SAMPLES,
    seed: int = 0,
) -> list[Path]:
    """Save `n_samples` random val-samples as GT | Pred side-by-side per channel.

    Parameters
    ----------
    predictions, targets : (N, Y, X, C) arrays, channels-last
    output_dir : Path to write PNGs to (created if missing)

    Returns paths of the saved PNG files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n = min(n_samples, predictions.shape[0])
    rng = np.random.default_rng(seed)
    indices = rng.choice(predictions.shape[0], size=n, replace=False)
    C = predictions.shape[-1]
    saved: list[Path] = []
    for i, idx in enumerate(indices):
        fig, axes = plt.subplots(C, 2, figsize=(6, 3 * C), squeeze=False)
        for ch in range(C):
            gt = targets[idx, ..., ch]
            pr = predictions[idx, ..., ch]
            psnr = _psnr(pr, gt)
            axes[ch, 0].imshow(gt, cmap="magma")
            axes[ch, 0].set_title(f"GT Ch{ch}")
            axes[ch, 0].axis("off")
            axes[ch, 1].imshow(pr, cmap="magma")
            axes[ch, 1].set_title(f"Pred Ch{ch}  PSNR={psnr:.2f} dB")
            axes[ch, 1].axis("off")
        fig.suptitle(f"val sample idx={int(idx)}")
        fig.tight_layout()
        out = output_dir / f"preview_idx{int(idx):03d}.png"
        fig.savefig(out, dpi=100)
        plt.close(fig)
        saved.append(out)
    return saved


def predict_and_eval(
    model: MicroSplitModule,
    trainer: Trainer,
    config: Any,
    dm: MicroSplitNgDataModule,
    exposure: str,
    data_root: Path,
    patch_size: tuple[int, int],
    output_metrics: str,
    preview_dir: Path,
) -> None:
    data_path = data_root / exposure
    _, _, va_in, _, _, _ = load_split_arrays(str(data_path), exposure, CH_IDX_LIST)
    pred_cfg = _predict_cfg(config, patch_size)
    dm.pred_config = pred_cfg
    dm.pred_input = [va_in]
    dm.setup("predict")
    predictions = trainer.predict(model, datamodule=dm)
    stitched, _ = convert_prediction(predictions, tiled=True, restore_shape=True)
    arr = np.concatenate(stitched, axis=0) if len(stitched) > 1 else stitched[0]
    if arr.ndim == 4 and arr.shape[1] == OUTPUT_CHANNELS:
        arr = np.moveaxis(arr, 1, -1)
    print(f"Predictions shape: {arr.shape}")

    # Evaluation against 500ms high-SNR reference
    highsnr_path = data_root / HIGHSNR_EXPOSURE_DURATION
    _, _, _, hs_target, _, _ = load_split_arrays(
        str(highsnr_path), HIGHSNR_EXPOSURE_DURATION, CH_IDX_LIST
    )
    hs_target_hwc = np.moveaxis(hs_target, 1, -1)
    metrics = compute_stats([hs_target_hwc], [arr])
    Path(output_metrics).parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics: {metrics}")

    saved = save_preview_images(arr, hs_target_hwc, preview_dir)
    print(f"Saved {len(saved)} preview PNGs to {preview_dir}")


def main(args) -> None:
    L.seed_everything(args.seed, workers=True)
    data_root = Path(args.data_root)
    exposure = EXPOSURE_DURATION
    data_path = data_root / exposure
    noise_model_dir = (
        BASE_DIR / "noise_models" / f"{exposure}_ngds_{args.experiment_name}"
    )
    output_dir = BASE_DIR / "lvae_checkpoints" / exposure / args.experiment_name
    output_metrics = (
        args.output_metrics or f"data/{args.experiment_name}/data/test/metrics.json"
    )
    preview_dir = (
        Path(args.preview_dir) if args.preview_dir else (output_dir / "previews")
    )

    # ------- multi-GPU resolution -------
    if args.global_batch_size is not None:
        if args.global_batch_size % args.devices != 0:
            raise ValueError(
                f"--global-batch-size ({args.global_batch_size}) must be divisible "
                f"by --devices ({args.devices})."
            )
        per_gpu_batch = args.global_batch_size // args.devices
    else:
        per_gpu_batch = args.batch_size
    strategy = args.strategy if args.strategy != "auto" else (
        "ddp" if args.devices > 1 else "auto"
    )
    sync_bn = (
        args.sync_batchnorm if args.sync_batchnorm is not None else args.devices > 1
    )

    # ------- noise model resolution -------
    if args.no_noise_model:
        nm_paths = None
    elif args.noise_model_paths is not None:
        nm_paths = [Path(p) for p in args.noise_model_paths]
    elif args.skip_training:
        # Predict-only: NM is loss-side, not needed for forward pass
        nm_paths = None
    else:
        # N2V + GMM fit is single-GPU (separate CAREamist Trainer inside).
        nm_input = prepare_noise_model_data(data_path, exposure)
        n2v_pred = train_n2v(
            nm_input,
            args.experiment_name,
            noise_model_dir,
            n2v_num_epochs=args.n2v_num_epochs,
            patch_size=tuple(args.patch_size),
            batch_size=per_gpu_batch,
            seed=args.seed,
        )
        nm_paths = fit_noise_models(n2v_pred, nm_input, noise_model_dir)

    # ------- config -------
    config = build_microsplit_config(
        experiment_name=args.experiment_name,
        nm_paths=nm_paths,
        num_epochs=args.num_epochs,
        batch_size=per_gpu_batch,
        num_workers=args.num_workers,
        patch_size=tuple(args.patch_size),
        use_wandb=not args.no_wandb,
        seed=args.seed,
        mmse_count=args.mmse_count,
    )

    # ------- data -------
    tr_in, tr_tg, va_in, va_tg, _, _ = load_split_arrays(
        str(data_path), exposure, CH_IDX_LIST
    )
    train_cfg, val_cfg = _train_val_cfgs(config)
    dm = MicroSplitNgDataModule(
        train_config=train_cfg,
        val_config=val_cfg,
        train_input=tr_in,
        train_target=tr_tg,
        val_input=va_in,
        val_target=va_tg,
        batch_size=per_gpu_batch,
        num_workers=args.num_workers,
    )

    # ------- model + trainer -------
    model = MicroSplitModule(config.algorithm_config)
    trainer = create_trainer(
        config,
        output_dir,
        args.experiment_name,
        devices=args.devices,
        strategy=strategy,
        sync_batchnorm=sync_bn,
    )

    if args.skip_training:
        if args.pretrained_ckpt is None:
            raise ValueError("--skip-training requires --pretrained-ckpt")
        load_pretrained_model(model, args.pretrained_ckpt)
        dm.setup("fit")
    else:
        if args.pretrained_ckpt is not None:
            load_pretrained_model(model, args.pretrained_ckpt)
        trainer.fit(model, datamodule=dm)

    # Non-rank-0 DDP workers exit here; predict + eval runs single-GPU on rank 0.
    if args.devices > 1:
        trainer.strategy.barrier()
    if not trainer.is_global_zero:
        return

    # Tear down the DDP process group before creating a fresh single-GPU Trainer:
    # otherwise Lightning "auto" resolves to DDP from lingering env vars, and the
    # first collective hangs against exited workers until the NCCL watchdog kills
    # the run. Also drop the launcher env vars that make PL redetect DDP.
    if args.devices > 1:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        for _k in (
            "LOCAL_RANK", "NODE_RANK", "WORLD_SIZE", "RANK",
            "MASTER_ADDR", "MASTER_PORT", "GROUP_RANK",
        ):
            os.environ.pop(_k, None)

    if not args.skip_predict:
        predict_trainer = (
            trainer
            if args.devices == 1
            else create_trainer(config, output_dir, args.experiment_name)
        )
        predict_and_eval(
            model,
            predict_trainer,
            config,
            dm,
            exposure,
            data_root,
            tuple(args.patch_size),
            output_metrics,
            preview_dir,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MicroSplit K=2 training on LIF24 5ms TIFF."
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Isolates checkpoints, noise-model dir, wandb run.",
    )

    # data
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help=f"Directory containing 5ms/ and 500ms/ Set1..6 subdirs. Default: {DEFAULT_DATA_ROOT}",
    )

    # patching
    parser.add_argument(
        "--patch-size", nargs=2, type=int, default=[64, 64], metavar=("Y", "X")
    )

    # training knobs
    parser.add_argument("--num-epochs", type=int, default=40)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Per-GPU batch size. Under DDP, global batch = batch_size × devices. "
        "Use --global-batch-size to pin the global batch instead.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n2v-num-epochs", type=int, default=10)
    parser.add_argument(
        "--mmse-count",
        type=int,
        default=MMSE_COUNT,
        help="Number of posterior samples averaged at predict time (MMSE).",
    )

    # multi-GPU
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of GPUs for Trainer.fit. 1 = current single-GPU behavior. "
        ">1 enables DDP. Predict + eval always runs single-GPU on rank 0.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
        help="Lightning strategy. 'auto' resolves to 'ddp' when devices>1. "
        "Escape hatch: 'ddp_find_unused_parameters_true'.",
    )
    parser.add_argument(
        "--sync-batchnorm",
        dest="sync_batchnorm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Sync BN running stats across ranks. Default: on when devices>1, off otherwise.",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=None,
        help="If set, per-GPU batch = global_batch_size // devices (overrides --batch-size). "
        "Use this to preserve the historical global batch of 64 across a DDP run.",
    )

    # noise model
    parser.add_argument(
        "--noise-model-paths",
        nargs="+",
        type=str,
        default=None,
        help="Pretrained GMM .npz paths (K=2). Skip N2V+GMM fit when supplied.",
    )
    parser.add_argument(
        "--no-noise-model",
        action="store_true",
        help="Pure µSplit: no NM, no N2V/GMM, sets loss weights musplit=1, denoisplit=0.",
    )

    # modes
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip Trainer.fit; predict-only. Requires --pretrained-ckpt.",
    )
    parser.add_argument(
        "--skip-predict",
        action="store_true",
        help="Skip predict + eval + preview images after training.",
    )
    parser.add_argument(
        "--pretrained-ckpt",
        type=str,
        default=None,
        help="LVAE checkpoint to load before training/predict.",
    )

    # output
    parser.add_argument(
        "--output-metrics", type=str, default=None, help="Where to write metrics.json."
    )
    parser.add_argument(
        "--preview-dir",
        type=str,
        default=None,
        help="Where to write 4 preview PNGs. Default: <output-dir>/previews/",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logger.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for torch/numpy/random and config seeding. "
        "Same seed → reproducible run (modulo cuDNN non-determinism).",
    )

    args = parser.parse_args()
    main(args)
