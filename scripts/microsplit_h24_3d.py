"""
MicroSplit training on the HT_H24 3D dataset (TIFF).

Source file (single):
  <data-root>/WTC11_WT_DIV25_3_1_0001.tif   shape (T=16, Z=69, C=3, Y=1608, X=1608) uint16
Data filter (before loading into pipeline):
  Z-slab [z_start:z_stop] + drop raw ch 2 → 15 Z-planes, 2 target channels.
Network input: synthesized as α·ch0 + (1-α)·ch1 (α=0.5 by default),
matching MultiChDloader(input_is_sum=False). Raw ch 2 is NOT used.

Examples
--------
Full training from scratch (fits N2V + noise models, then trains LVAE):

    python scripts/microsplit_h24_3d.py \\
        --experiment-name ht_h24_3d_ngds_50ep --num-epochs 50

Default with pretrained noise models:

    python scripts/microsplit_h24_3d.py \\
        --experiment-name ht_h24_3d_ngds_50ep \\
        --noise-model-paths scripts/noise_models/h24_3d_ngds_<EXP>/noise_model_Ch{0,1}.npz

Predict-only from a checkpoint:

    python scripts/microsplit_h24_3d.py \\
        --experiment-name ht_h24_3d_ngds_50ep --skip-training \\
        --pretrained-ckpt scripts/lvae_checkpoints/h24_3d/<EXP>/checkpoints/last.ckpt
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
DEFAULT_DATA_ROOT = Path("/group/jug/public_html/microsplit/ht_h24_3d_tiff")
DATA_FILE = "WTC11_WT_DIV25_3_1_0001.tif"

Z_START = 25
Z_STOP = 40
CH_IDX_LIST = [0, 1]  # 2 targets; input is synthesized

BASE_DIR = Path(__file__).resolve().parent

PATCH_SIZE = (9, 64, 64)
ENCODER_CONV_STRIDES = [1, 2, 2]
DECODER_CONV_STRIDES = [1, 2, 2]
Z_DIMS = [128] * 4
N_FILTERS = 64  # post PR #1049: single field, applies to both encoder and decoder
MULTISCALE_COUNT = 1
OUTPUT_CHANNELS = len(CH_IDX_LIST)  # 2
MMSE_COUNT = 1
GRID_YX = 32
LR = 1e-3

N2V_PATCH_SIZE = (8, 64, 64)
N2V_PREDICT_TILE_SIZE = (8, 128, 128)
N2V_PREDICT_TILE_OVERLAP = (0, 32, 32)

TRAINER_PRECISION = 16
TRAINER_GRADIENT_CLIP_VAL = 0.5
TRAINER_GRADIENT_CLIP_ALGORITHM = "value"

NM_N_GAUSSIAN = 3
NM_N_COEFF = 3
NM_MIN_SIGMA = 200.0
NM_N_EPOCHS = 2000

VAL_FRACTION = 0.1
TEST_FRACTION = 0.1

N_PREVIEW_SAMPLES = 4


# ---------------------------------------------------------------------------
# Data filter + loading
# ---------------------------------------------------------------------------


def filter_ht_h24_volume(
    volume: np.ndarray,
    z_start: int = Z_START,
    z_stop: int = Z_STOP,
    channel_list: list[int] = CH_IDX_LIST,
) -> np.ndarray:
    """Notebook data-space filter: Z-slab + channel selection.

    volume : (T, Z, C, Y, X) uint16 → (T, Z', C', Y, X) uint16
    """
    if volume.ndim != 5:
        raise ValueError(f"expected 5D (T,Z,C,Y,X), got shape {volume.shape}")
    return volume[:, z_start:z_stop, channel_list, :, :]


def load_filtered_volume(
    data_file: Path | str,
    z_start: int = Z_START,
    z_stop: int = Z_STOP,
    channel_list: list[int] = CH_IDX_LIST,
) -> np.ndarray:
    volume = tifffile.imread(str(data_file))
    return filter_ht_h24_volume(
        volume, z_start=z_start, z_stop=z_stop, channel_list=channel_list
    )


def synthesize_input(target_arr: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Synthesize noisy MicroSplit input as α·ch0 + (1-α)·ch1."""
    if target_arr.shape[1] != 2:
        raise ValueError(f"expected 2 target channels, got {target_arr.shape[1]}")
    return (alpha * target_arr[:, 0:1] + (1.0 - alpha) * target_arr[:, 1:2]).astype(
        target_arr.dtype
    )


def load_split_arrays(
    data_file: Path | str,
    z_start: int,
    z_stop: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    filtered = load_filtered_volume(data_file, z_start, z_stop).astype(np.float32)
    scyzx = np.moveaxis(filtered, 2, 1)  # (T, C, Z, Y, X)
    train_idx, val_idx, test_idx = get_datasplit_tuples(
        VAL_FRACTION, TEST_FRACTION, scyzx.shape[0]
    )
    target_arr = scyzx
    input_arr = synthesize_input(target_arr, alpha=alpha)
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


class MicroSplitNgDataModule3D(L.LightningDataModule):
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
        batch_size: int = 32,
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

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=default_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
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


def prepare_noise_model_data(data_file: Path, z_start: int, z_stop: int) -> np.ndarray:
    filtered = load_filtered_volume(data_file, z_start, z_stop).astype(np.float32)
    train_idx, _, _ = get_datasplit_tuples(
        VAL_FRACTION, TEST_FRACTION, filtered.shape[0]
    )
    return np.moveaxis(filtered[train_idx], 2, -1)  # (T', Z, Y, X, C)


def train_n2v(
    nm_input: np.ndarray,
    experiment_name: str,
    noise_model_dir: Path,
    n2v_num_epochs: int,
    batch_size: int,
    seed: int,
    predict_tile_size: tuple[int, ...] = N2V_PREDICT_TILE_SIZE,
    predict_tile_overlap: tuple[int, ...] = N2V_PREDICT_TILE_OVERLAP,
) -> np.ndarray:
    config = create_n2v_config(
        experiment_name=f"{experiment_name}_n2v",
        data_type="array",
        axes="SZYXC",
        n_channels=len(CH_IDX_LIST),
        patch_size=N2V_PATCH_SIZE,
        batch_size=batch_size,
        num_epochs=n2v_num_epochs,
    )
    config.data_config.seed = seed
    careamist = CAREamist(config=config, work_dir=str(noise_model_dir))
    careamist.train(train_data=nm_input)
    prediction, _ = careamist.predict(
        nm_input,
        tile_size=predict_tile_size,
        tile_overlap=predict_tile_overlap,
    )
    return np.concatenate(prediction, axis=0)


def fit_noise_models(
    signal: np.ndarray,
    observation: np.ndarray,
    noise_model_dir: Path,
) -> list[Path]:
    noise_model_dir.mkdir(parents=True, exist_ok=True)
    trainer = NoiseModelTrainer(
        n_gaussian=NM_N_GAUSSIAN,
        n_coeff=NM_N_COEFF,
        min_sigma=NM_MIN_SIGMA,
    )
    trainer.train_from_pairs(
        signal=signal,
        observation=observation,
        signal_axes="SZYXC",
        observation_axes="SZYXC",
        n_epochs=NM_N_EPOCHS,
    )
    return trainer.save(noise_model_dir)


# ---------------------------------------------------------------------------
# MicroSplit config
# ---------------------------------------------------------------------------


def build_microsplit_config(
    experiment_name: str,
    nm_paths: Optional[list[Path]],
    num_epochs: int,
    batch_size: int,
    num_workers: int,
    use_wandb: bool,
    seed: int,
    mmse_count: int = MMSE_COUNT,
) -> Any:
    if nm_paths:
        nm_config = NoiseModelTrainer.config_from_paths(nm_paths)
        musplit_w, denoisplit_w = 0.1, 0.9
    else:
        nm_config = None
        musplit_w, denoisplit_w = 1.0, 0.0

    return create_advanced_microsplit_config(
        experiment_name=experiment_name,
        data_type="array",
        axes="SCZYX",
        patch_size=PATCH_SIZE,
        output_channels=OUTPUT_CHANNELS,
        multiscale_count=MULTISCALE_COUNT,
        batch_size=batch_size,
        num_epochs=num_epochs,
        mmse_count=mmse_count,
        noise_model=nm_config,
        musplit_weight=musplit_w,
        denoisplit_weight=denoisplit_w,
        augmentations=[],
        encoder_conv_strides=ENCODER_CONV_STRIDES,
        decoder_conv_strides=DECODER_CONV_STRIDES,
        model_params={
            "z_dims": Z_DIMS,
            "n_filters": N_FILTERS,
        },
        optimizer_params={"lr": LR, "weight_decay": 0},
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
                filename=f"ht_h24_3d_{experiment_name}",
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
        )
    return Trainer(**kwargs)


def _train_val_cfgs(config: Any) -> tuple[MicroSplitDataConfig, MicroSplitDataConfig]:
    train_cfg: MicroSplitDataConfig = config.data_config
    return train_cfg, train_cfg.convert_mode("validating")


def _predict_cfg(config: Any) -> MicroSplitDataConfig:
    train_cfg: MicroSplitDataConfig = config.data_config
    overlap = (0, PATCH_SIZE[1] - GRID_YX, PATCH_SIZE[2] - GRID_YX)
    return train_cfg.convert_mode(
        "predicting",
        new_patch_size=PATCH_SIZE,
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
    """Save `n_samples` random val-samples as GT | Pred per channel (mid-Z slice)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n = min(n_samples, predictions.shape[0])
    rng = np.random.default_rng(seed)
    indices = rng.choice(predictions.shape[0], size=n, replace=False)
    C = predictions.shape[-1]
    z_mid = predictions.shape[1] // 2
    saved: list[Path] = []
    for idx in indices:
        fig, axes = plt.subplots(C, 2, figsize=(6, 3 * C), squeeze=False)
        for ch in range(C):
            gt = targets[idx, z_mid, ..., ch]
            pr = predictions[idx, z_mid, ..., ch]
            psnr = _psnr(pr, gt)
            axes[ch, 0].imshow(gt, cmap="magma")
            axes[ch, 0].set_title(f"GT Ch{ch}")
            axes[ch, 0].axis("off")
            axes[ch, 1].imshow(pr, cmap="magma")
            axes[ch, 1].set_title(f"Pred Ch{ch}  PSNR={psnr:.2f} dB")
            axes[ch, 1].axis("off")
        fig.suptitle(f"val sample idx={int(idx)}  z={z_mid}")
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
    dm: MicroSplitNgDataModule3D,
    data_file: Path,
    batch_size: int,
    z_start: int,
    z_stop: int,
    alpha: float,
    output_metrics: str,
    preview_dir: Path,
) -> None:
    _, _, va_in, va_tg, _, _ = load_split_arrays(data_file, z_start, z_stop, alpha)
    pred_cfg = _predict_cfg(config)
    dm.pred_config = pred_cfg
    dm.pred_input = [va_in]
    dm.batch_size = batch_size
    dm.setup("predict")
    predictions = trainer.predict(model, datamodule=dm)
    stitched, _ = convert_prediction(predictions, tiled=True, restore_shape=True)
    arr = np.concatenate(stitched, axis=0) if len(stitched) > 1 else stitched[0]
    if arr.ndim == 5 and arr.shape[1] == OUTPUT_CHANNELS:
        arr = np.moveaxis(arr, 1, -1)  # (S, Z, Y, X, C)
    print(f"Predictions shape: {arr.shape}")

    target_hwc = np.moveaxis(va_tg, 1, -1)  # (S, Z, Y, X, C)
    # Flatten Z into batch dim for compute_stats (2D-per-slice)
    if arr.ndim == 5:
        s, z, y, x, c = arr.shape
        arr_2d = arr.reshape(s * z, y, x, c)
        tg_2d = target_hwc.reshape(s * z, y, x, c)
    else:
        arr_2d, tg_2d = arr, target_hwc
    metrics = compute_stats([tg_2d], [arr_2d])
    Path(output_metrics).parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics: {metrics}")

    saved = save_preview_images(arr, target_hwc, preview_dir)
    print(f"Saved {len(saved)} preview PNGs to {preview_dir}")


def main(args) -> None:
    L.seed_everything(args.seed, workers=True)
    data_root = Path(args.data_root)
    data_file = data_root / DATA_FILE
    noise_model_dir = BASE_DIR / "noise_models" / f"h24_3d_ngds_{args.experiment_name}"
    output_dir = BASE_DIR / "lvae_checkpoints" / "h24_3d" / args.experiment_name
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

    if args.no_noise_model:
        nm_paths = None
    elif args.noise_model_paths is not None:
        nm_paths = [Path(p) for p in args.noise_model_paths]
    elif args.skip_training:
        nm_paths = None
    else:
        nm_input = prepare_noise_model_data(data_file, args.z_start, args.z_stop)
        n2v_pred = train_n2v(
            nm_input,
            args.experiment_name,
            noise_model_dir,
            n2v_num_epochs=args.n2v_num_epochs,
            batch_size=per_gpu_batch,
            seed=args.seed,
            predict_tile_size=tuple(args.n2v_predict_tile_size),
            predict_tile_overlap=tuple(args.n2v_predict_tile_overlap),
        )
        nm_paths = fit_noise_models(n2v_pred, nm_input, noise_model_dir)

    config = build_microsplit_config(
        experiment_name=args.experiment_name,
        nm_paths=nm_paths,
        num_epochs=args.num_epochs,
        batch_size=per_gpu_batch,
        num_workers=args.num_workers,
        use_wandb=not args.no_wandb,
        seed=args.seed,
        mmse_count=args.mmse_count,
    )

    tr_in, tr_tg, va_in, va_tg, _, _ = load_split_arrays(
        data_file,
        args.z_start,
        args.z_stop,
        args.alpha,
    )
    train_cfg, val_cfg = _train_val_cfgs(config)
    dm = MicroSplitNgDataModule3D(
        train_config=train_cfg,
        val_config=val_cfg,
        train_input=tr_in,
        train_target=tr_tg,
        val_input=va_in,
        val_target=va_tg,
        batch_size=per_gpu_batch,
        num_workers=args.num_workers,
    )

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
            data_file,
            per_gpu_batch,
            args.z_start,
            args.z_stop,
            args.alpha,
            output_metrics,
            preview_dir,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MicroSplit 3D training on HT_H24 TIFF."
    )
    parser.add_argument("--experiment-name", type=str, required=True)

    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help=f"Directory containing {DATA_FILE}. Default: {DEFAULT_DATA_ROOT}",
    )

    parser.add_argument("--z-start", type=int, default=Z_START)
    parser.add_argument(
        "--z-stop",
        type=int,
        default=Z_STOP,
        help="exclusive; default keeps 15 planes [25:40]",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Input synthesis weight: input = α·ch0 + (1-α)·ch1",
    )

    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
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
        "Use this to preserve the historical global batch of 32 across a DDP run.",
    )

    parser.add_argument(
        "--n2v-predict-tile-size",
        nargs=3,
        type=int,
        default=list(N2V_PREDICT_TILE_SIZE),
        metavar=("Z", "Y", "X"),
    )
    parser.add_argument(
        "--n2v-predict-tile-overlap",
        nargs=3,
        type=int,
        default=list(N2V_PREDICT_TILE_OVERLAP),
        metavar=("Z", "Y", "X"),
    )

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

    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip Trainer.fit; predict-only. Requires --pretrained-ckpt.",
    )
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--pretrained-ckpt", type=str, default=None)

    parser.add_argument("--output-metrics", type=str, default=None)
    parser.add_argument(
        "--preview-dir",
        type=str,
        default=None,
        help="Where to write 4 preview PNGs. Default: <output-dir>/previews/",
    )
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for torch/numpy/random and config seeding. "
        "Same seed → reproducible run (modulo cuDNN non-determinism).",
    )

    args = parser.parse_args()
    main(args)
