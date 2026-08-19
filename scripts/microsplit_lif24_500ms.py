"""
MicroSplit training on the LIF24 3-channel 500ms dataset (K=3 unmixing, TIFF).

Channel layout (remapped in the pre-extracted 4-ch TIFF):
0 = MicroTubules
1 = NuclearMembrane
2 = Centromere
3 = superimposed "123"

Examples
--------
Full training from scratch (fits N2V + noise models, then trains LVAE):

    python scripts/microsplit_lif24_500ms.py \\
        --experiment-name ht_lif24_500ms_ngds_nm

Pure µSplit (no NM, no N2V/GMM):

    python scripts/microsplit_lif24_500ms.py \\
        --experiment-name ht_lif24_500ms_ngds_musplit --no-noise-model

Predict-only from a checkpoint:

    python scripts/microsplit_lif24_500ms.py \\
        --experiment-name ht_lif24_500ms_ngds_nm \\
        --skip-training \\
        --pretrained-ckpt scripts/lvae_checkpoints/lif24_500ms/<EXP>/checkpoints/last.ckpt
"""

from __future__ import annotations

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
DEFAULT_DATA_ROOT = Path("/group/jug/public_html/microsplit/ht_lif24_3ch_500ms_tiff")
CH_IDX_LIST = [0, 1, 2, 3]  # 3 targets (0,1,2) + superimposed input (3)

BASE_DIR = Path(__file__).resolve().parent

Z_DIMS = [128] * 4
N_FILTERS = 64   # post PR #1049: single field, applies to both encoder and decoder
MULTISCALE_COUNT = 3
OUTPUT_CHANNELS = len(CH_IDX_LIST) - 1  # 3
MMSE_COUNT = 1
GRID_SIZE = 32
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
# TIFF loading
# ---------------------------------------------------------------------------


def _datafiles() -> list[str]:
    return [f"Set{i}/uSplit_500ms.tif" for i in range(1, 7)]


def _load_one_fpath(fpath: str, channel_list: list[int]) -> np.ndarray:
    data = tifffile.imread(fpath)  # (P, C, Y, X) uint16
    data = data[:, channel_list, ...]
    return np.moveaxis(data, 1, -1)  # (P, Y, X, C)


def _load_data(datadir: str, channel_list: list[int]) -> np.ndarray:
    return np.concatenate(
        [_load_one_fpath(os.path.join(datadir, f), channel_list) for f in _datafiles()],
        axis=0,
    )


def _syxc_to_scyx(arr: np.ndarray) -> np.ndarray:
    return np.moveaxis(arr, -1, 1)


def load_split_arrays(
    datadir: str, channel_list: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = _load_data(datadir, channel_list).astype(np.float32)
    train_idx, val_idx, test_idx = get_datasplit_tuples(
        VAL_FRACTION, TEST_FRACTION, len(data)
    )
    input_arr = _syxc_to_scyx(data[..., -1:])  # (N, 1, Y, X)
    target_arr = _syxc_to_scyx(data[..., :-1])  # (N, 3, Y, X)
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


def prepare_noise_model_data(data_path: Path) -> np.ndarray:
    data = _load_data(str(data_path), CH_IDX_LIST).astype(np.float32)
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
    noise_model_dir.mkdir(parents=True, exist_ok=True)
    signal_targets = signal[..., :-1]
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
    experiment_name: str, nm_paths: Optional[list[Path]], num_epochs: int,
    batch_size: int, num_workers: int, patch_size: tuple[int, int],
    use_wandb: bool, seed: int, mmse_count: int = MMSE_COUNT,
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
        axes="SCYX",
        patch_size=patch_size,
        output_channels=OUTPUT_CHANNELS,
        multiscale_count=MULTISCALE_COUNT,
        batch_size=batch_size, num_epochs=num_epochs,
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


def create_trainer(config: Any, output_dir: Path, experiment_name: str) -> Trainer:
    output_dir.mkdir(parents=True, exist_ok=True)
    return Trainer(
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
                filename=f"ht_lif24_500ms_3ch_{experiment_name}",
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
    """Save `n_samples` random val-samples as GT | Pred per channel."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n = min(n_samples, predictions.shape[0])
    rng = np.random.default_rng(seed)
    indices = rng.choice(predictions.shape[0], size=n, replace=False)
    C = predictions.shape[-1]
    saved: list[Path] = []
    for idx in indices:
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
    data_path: Path,
    patch_size: tuple[int, int],
    output_metrics: str,
    preview_dir: Path,
) -> None:
    _, _, va_in, va_tg, _, _ = load_split_arrays(str(data_path), CH_IDX_LIST)
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

    target_hwc = np.moveaxis(va_tg, 1, -1)
    metrics = compute_stats([target_hwc], [arr])
    Path(output_metrics).parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics: {metrics}")

    saved = save_preview_images(arr, target_hwc, preview_dir)
    print(f"Saved {len(saved)} preview PNGs to {preview_dir}")


def main(args) -> None:
    L.seed_everything(args.seed, workers=True)
    data_path = Path(args.data_root)
    noise_model_dir = (
        BASE_DIR / "noise_models" / f"lif24_500ms_ngds_{args.experiment_name}"
    )
    output_dir = BASE_DIR / "lvae_checkpoints" / "lif24_500ms" / args.experiment_name
    output_metrics = (
        args.output_metrics or f"data/{args.experiment_name}/data/test/metrics.json"
    )
    preview_dir = (
        Path(args.preview_dir) if args.preview_dir else (output_dir / "previews")
    )

    if args.no_noise_model:
        nm_paths = None
    elif args.noise_model_paths is not None:
        nm_paths = [Path(p) for p in args.noise_model_paths]
    elif args.skip_training:
        nm_paths = None
    else:
        nm_input = prepare_noise_model_data(data_path)
        n2v_pred = train_n2v(
            nm_input,
            args.experiment_name,
            noise_model_dir,
            n2v_num_epochs=args.n2v_num_epochs,
            patch_size=tuple(args.patch_size),
            batch_size=args.batch_size,
            seed=args.seed,
        )
        nm_paths = fit_noise_models(n2v_pred, nm_input, noise_model_dir)

    config = build_microsplit_config(
        experiment_name=args.experiment_name,
        nm_paths=nm_paths,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=tuple(args.patch_size),
        use_wandb=not args.no_wandb,
        seed=args.seed,
        mmse_count=args.mmse_count,
    )

    tr_in, tr_tg, va_in, va_tg, _, _ = load_split_arrays(str(data_path), CH_IDX_LIST)
    train_cfg, val_cfg = _train_val_cfgs(config)
    dm = MicroSplitNgDataModule(
        train_config=train_cfg,
        val_config=val_cfg,
        train_input=tr_in,
        train_target=tr_tg,
        val_input=va_in,
        val_target=va_tg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = MicroSplitModule(config.algorithm_config)
    trainer = create_trainer(config, output_dir, args.experiment_name)

    if args.skip_training:
        if args.pretrained_ckpt is None:
            raise ValueError("--skip-training requires --pretrained-ckpt")
        load_pretrained_model(model, args.pretrained_ckpt)
        dm.setup("fit")
    else:
        if args.pretrained_ckpt is not None:
            load_pretrained_model(model, args.pretrained_ckpt)
        trainer.fit(model, datamodule=dm)

    if not args.skip_predict:
        predict_and_eval(
            model,
            trainer,
            config,
            dm,
            data_path,
            tuple(args.patch_size),
            output_metrics,
            preview_dir,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MicroSplit K=3 training on LIF24 3-channel 500ms TIFF."
    )
    parser.add_argument("--experiment-name", type=str, required=True)

    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help=f"Directory containing Set1..6/uSplit_500ms.tif. Default: {DEFAULT_DATA_ROOT}",
    )

    parser.add_argument(
        "--patch-size", nargs=2, type=int, default=[64, 64], metavar=("Y", "X")
    )

    parser.add_argument("--num-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n2v-num-epochs", type=int, default=10)
    parser.add_argument("--mmse-count", type=int, default=MMSE_COUNT,
                        help="Number of posterior samples averaged at predict time (MMSE).")

    parser.add_argument("--noise-model-paths", nargs="+", type=str, default=None,
                        help="Pretrained GMM .npz paths (K=3). Skip N2V+GMM fit when supplied.")
    parser.add_argument("--no-noise-model", action="store_true",
                        help="Pure µSplit: no NM, no N2V/GMM, sets loss weights musplit=1, denoisplit=0.")

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
