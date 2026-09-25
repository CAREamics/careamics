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
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import lightning.pytorch as L
import tifffile
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
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
from careamics.lightning.modules.module_utils import request_model_compilation
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
# Names every output directory and checkpoint. Equals the exposure for the
# default 2-split so existing 5ms paths are unchanged; `configure_dataset` makes
# it e.g. "3split_2ms" for the other published variants.
DATASET_TAG = EXPOSURE_DURATION

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


def configure_dataset(exposure: str, ch_idx_list: list[int]) -> None:
    """Select the HT_LIF24 variant: exposure and raw channels (targets..., input).

    The published variants (disentangle `2406/D25-M3-S0-L8/*`) differ only in
    these two, so they are set once here instead of threading them through
    every loader. Raw channel indices follow `NikolaChannelList`: 0-3 = A-D,
    8 = AB, 17 = BCD, 18 = ABCD. The last entry is always the real
    superimposed input channel (training Mode III).
    """
    global EXPOSURE_DURATION, CH_IDX_LIST, OUTPUT_CHANNELS, DATASET_TAG
    EXPOSURE_DURATION = exposure
    CH_IDX_LIST = list(ch_idx_list)
    OUTPUT_CHANNELS = len(CH_IDX_LIST) - 1
    default = exposure == "5ms" and CH_IDX_LIST == [0, 1, 8]
    DATASET_TAG = exposure if default else f"{OUTPUT_CHANNELS}split_{exposure}"


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
    # --denoise-input: swap the noisy superimposed input for its N2V denoising.
    # Guarded on exposure so the 500ms high-SNR load (which only uses targets)
    # is never touched, and on length so a Set subset cannot silently misalign.
    if (
        _DENOISED_INPUT is not None
        and exposure == EXPOSURE_DURATION
        and len(_DENOISED_INPUT) == len(input_arr)
    ):
        input_arr = _DENOISED_INPUT.astype(np.float32)
        print(
            f"[BRANCH] load_split_arrays(exposure={exposure!r}): USING "
            f"N2V-DENOISED input {input_arr.shape}"
        )
    else:
        print(
            f"[BRANCH] load_split_arrays(exposure={exposure!r}): using RAW input "
            f"{input_arr.shape} from raw channel {channel_list[-1]}  "
            f"[_DENOISED_INPUT={'set' if _DENOISED_INPUT is not None else 'None'}, "
            f"exposure_match={exposure == EXPOSURE_DURATION}]"
        )
    target_arr = _syxc_to_scyx(data[..., :-1])  # (N, 2, Y, X)
    return (
        input_arr[train_idx],
        target_arr[train_idx],
        input_arr[val_idx],
        target_arr[val_idx],
        input_arr[test_idx],
        target_arr[test_idx],
    )


def load_sets_arrays(
    datadir: str, exposure: str, channel_list: list[int], sets: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """All frames of the given Sets, unsplit -> (input, target) SCYX arrays.

    Frame k of the returned arrays is frame k of the Sets' TIFF concatenation,
    so outputs stay directly indexable against the raw files (used by
    `--predict-sets`, matching the reference-pipeline bundle from
    `microsplit_lif24_5ms_reference.py`).
    """
    data = np.concatenate(
        [
            _load_one_fpath(
                os.path.join(datadir, s, f"uSplit_{exposure}.tif"), channel_list
            )
            for s in sets
        ],
        axis=0,
    ).astype(np.float32)
    return _syxc_to_scyx(data[..., -1:]), _syxc_to_scyx(data[..., :-1])


def _set_frame_counts(datadir: str, exposure: str) -> dict[str, int]:
    """Frame count per Set from TIFF headers only (no pixel reads)."""
    counts = {}
    for f in _datafiles(exposure):
        with tifffile.TiffFile(os.path.join(datadir, f)) as tf:
            counts[f.split("/")[0]] = int(tf.series[0].shape[0])
    return counts


def _set_offsets(counts: dict[str, int]) -> tuple[list[str], dict[str, int], int]:
    """Set order, each Set's first global frame index, and the total count."""
    order = [f.split("/")[0] for f in _datafiles(EXPOSURE_DURATION)]
    offsets, acc = {}, 0
    for s in order:
        offsets[s] = acc
        acc += counts[s]
    return order, offsets, acc


def _frames_for_split(
    data_root: Path, exposure: str, split: str
) -> list[tuple[str, int, int]]:
    """(set, frame_idx_in_set, global_frame_idx) for one `get_datasplit_tuples` split.

    The global index is the position in the Set1..Set6 TIFF concatenation, i.e.
    exactly what `load_split_arrays` indexes, so slot k of the predictions is
    frame k of this list.
    """
    counts = _set_frame_counts(str(data_root / exposure), exposure)
    order, offsets, total = _set_offsets(counts)
    train_idx, val_idx, test_idx = get_datasplit_tuples(
        VAL_FRACTION, TEST_FRACTION, total
    )
    idx = {"train": train_idx, "val": val_idx, "canonical-test": test_idx}[split]
    frames = []
    for g in (int(i) for i in idx):
        for s in order:
            if offsets[s] <= g < offsets[s] + counts[s]:
                frames.append((s, g - offsets[s], g))
                break
    return frames


def _frames_for_sets(
    data_root: Path, exposure: str, sets: list[str]
) -> list[tuple[str, int, int]]:
    """(set, frame_idx_in_set, global_frame_idx) for every frame of `sets`."""
    counts = _set_frame_counts(str(data_root / exposure), exposure)
    _, offsets, _ = _set_offsets(counts)
    return [
        (s, k, offsets[s] + k) for s in sets for k in range(counts[s])
    ]


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
    n_gaussian: int = NM_N_GAUSSIAN,
    n_coeff: int = NM_N_COEFF,
    min_sigma: float = NM_MIN_SIGMA,
    n_epochs: int = NM_N_EPOCHS,
    global_signal_range: bool = False,
) -> list[Path]:
    """Fit one GMM per target channel via `NoiseModelTrainer.train_from_pairs`.

    Both `signal` (N2V predictions) and `observation` (raw noisy) are SYXC.
    Only target channels (all but last, which is the input) are used.
    """
    noise_model_dir.mkdir(parents=True, exist_ok=True)
    # drop the trailing input channel — fit NM only for target channels
    signal_targets = signal[..., :-1]  # (S, Y, X, C_out)
    observation_targets = observation[..., :-1]

    print(
        f"Fitting noise models: n_gaussian={n_gaussian} n_coeff={n_coeff} "
        f"min_sigma={min_sigma} n_epochs={n_epochs} "
        f"global_signal_range={global_signal_range}"
    )
    trainer = NoiseModelTrainer(
        n_gaussian=n_gaussian,
        n_coeff=n_coeff,
        min_sigma=min_sigma,
        global_signal_range=global_signal_range,
    )
    trainer.train_from_pairs(
        signal=signal_targets,
        observation=observation_targets,
        signal_axes="SYXC",
        observation_axes="SYXC",
        n_epochs=n_epochs,
    )
    report_noise_model_diagnostics(trainer)
    paths = trainer.save(noise_model_dir)
    (noise_model_dir / "nm_params.json").write_text(
        json.dumps(
            {
                "n_gaussian": n_gaussian,
                "n_coeff": n_coeff,
                "min_sigma": min_sigma,
                "n_epochs": n_epochs,
                "global_signal_range": global_signal_range,
            },
            indent=2,
        )
    )
    return paths


def report_noise_model_diagnostics(trainer: NoiseModelTrainer) -> None:
    """Print how much of the signal range the variance clamp eats, per channel.

    `min_sigma` clamps the VARIANCE polynomial, not sigma
    (`GaussianMixtureNoiseModel.get_gaussian_parameters` does
    ``sigma = sqrt(clamp(poly, min=min_sigma))``). Where the clamp is active the
    variance branch gets no gradient, so its coefficients stay at their
    initialisation -- `w[K:2K, 1] == log(max_signal - min_signal)`. If that
    happens over the whole signal range the GMM silently degenerates into a
    homoscedastic Gaussian with sigma = sqrt(min_sigma), and nothing in the fit
    warns about it. These lines make that visible.
    """
    if trainer.noise_models is None:
        return
    print("\n=== noise model diagnostics ===")
    for idx, nm in enumerate(trainer.noise_models):
        lo = float(nm.min_signal.item())
        hi = float(nm.max_signal.item())
        ms = float(nm.min_sigma.item())
        k = nm.n_gaussian
        grid = torch.linspace(lo, hi, 512)
        clamped, sig_lo, sig_hi = [], [], []
        for g in range(k):
            poly = nm.polynomial_regressor(
                torch.exp(nm.weight[k + g, :].detach()), grid
            )
            clamped.append(float((poly < ms).float().mean().item()))
            sigma = torch.sqrt(torch.clamp(poly, min=ms))
            sig_lo.append(float(sigma.min().item()))
            sig_hi.append(float(sigma.max().item()))
        init = float(np.log(hi - lo))
        drift = float((nm.weight[k : 2 * k, 1].detach() - init).abs().max().item())
        losses = trainer.train_losses[idx] if trainer.train_losses else []
        final = losses[-1] if losses else float("nan")
        pct = ", ".join(f"{c * 100:.1f}%" for c in clamped)
        print(
            f"  ch{idx}: signal [{lo:.4f}, {hi:.4f}]  min_sigma(variance)={ms:g}"
            f"  -> sigma floor {ms ** 0.5:.4f}"
        )
        print(
            f"        variance clamped over {min(clamped) * 100:.1f}-"
            f"{max(clamped) * 100:.1f}% of the signal range (per gaussian: {pct})"
        )
        print(
            f"        learned sigma range [{min(sig_lo):.4f}, {max(sig_hi):.4f}]"
        )
        print(
            f"        max |w[K:2K,1] - init| = {drift:.3e}"
            f"   (0 => variance branch never moved off init)"
        )
        print(f"        final fit NLL = {final:.5f}")
    print("=== end noise model diagnostics ===\n")


# ---------------------------------------------------------------------------
# Input denoising (alternative to the loss-side noise model)
# ---------------------------------------------------------------------------

# Set by main() when --denoise-input is passed; consumed by load_split_arrays.
_DENOISED_INPUT: Optional[np.ndarray] = None


def build_denoised_input(
    data_path: Path,
    exposure: str,
    experiment_name: str,
    work_dir: Path,
    n2v_num_epochs: int,
    patch_size: tuple[int, int],
    batch_size: int,
    seed: int,
    cache: Optional[Path],
) -> np.ndarray:
    """Return the N2V denoising of the input channel, as (N, 1, Y, X) SCYX.

    N2V is trained on the TRAIN split only and then applied to every frame, so
    the val/test inputs are denoised by a model that never saw them. N2V is
    self-supervised and never touches the 500ms reference, so this adds no
    supervision the noise-model pipeline did not already use -- it is the same
    N2V pass, redirected from fitting a GMM to cleaning the input.
    """
    if cache is not None and cache.exists():
        print(f"Loading cached denoised input from {cache}")
        return np.load(cache)

    data = _load_data(str(data_path), CH_IDX_LIST, exposure).astype(np.float32)
    train_idx, _, _ = get_datasplit_tuples(VAL_FRACTION, TEST_FRACTION, len(data))

    config = create_n2v_config(
        experiment_name=f"{experiment_name}_n2v_input",
        data_type="array",
        axes="SYXC",
        n_channels=len(CH_IDX_LIST),
        patch_size=patch_size,
        batch_size=batch_size,
        num_epochs=n2v_num_epochs,
    )
    config.data_config.seed = seed
    careamist = CAREamist(config=config, work_dir=str(work_dir))
    careamist.train(train_data=data[train_idx])
    prediction, _ = careamist.predict(data, tile_size=(256, 256))
    pred = np.concatenate(prediction, axis=0)  # (N, Y, X, C) SYXC
    denoised = _syxc_to_scyx(pred[..., -1:])  # input channel only, (N, 1, Y, X)

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, denoised)
        print(f"Cached denoised input to {cache}")
    return denoised


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
    denoisplit_weight: float = 0.9,
) -> Any:
    if nm_paths:
        # denoiSplit-heavy by default (factory default 0.9/0.1); the split is
        # exposed because it only makes sense to trust the noise-model
        # likelihood as far as the noise model is actually informative.
        denoisplit_w = denoisplit_weight
        musplit_w = 1.0 - denoisplit_weight
    else:
        musplit_w, denoisplit_w = 1.0, 0.0  # pure µSplit
    print(f"Loss weights: musplit={musplit_w} denoisplit={denoisplit_w}")

    config = create_advanced_microsplit_config(
        experiment_name=experiment_name,
        data_type="array",
        axes="SCYX",
        patch_size=patch_size,
        output_channels=OUTPUT_CHANNELS,
        multiscale_count=MULTISCALE_COUNT,
        batch_size=batch_size,
        num_epochs=num_epochs,
        gaussian_likelihood_weight=musplit_w,
        noise_model_likelihood_weight=denoisplit_w,
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
    """Load LVAE weights, accepting both checkpoint layouts.

    - `model.*`-prefixed keys => a MicroSplitModule checkpoint written by this
      script (careamics Lightning API).
    - bare LVAE keys          => the original / microSplit-reproducibility
      published format (`best_0_1.ckpt` == the original run's
      `BaselineVAECL_best.ckpt`). Loss-side keys (noise model / likelihood
      buffers) are dropped: they are not part of the forward pass and careamics
      does not build them. Everything else is re-prefixed with `model.` so a
      key that genuinely drifted still shows up as missing/unexpected below.
    """
    device = get_device()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if any(k.startswith("model.") for k in state):
        layout = "careamics MicroSplitModule"
    else:
        loss_side = ("noiseModel", "noise_model", "likelihood")
        dropped = sorted(k for k in state if k.startswith(loss_side))
        state = {
            f"model.{k}": v for k, v in state.items() if not k.startswith(loss_side)
        }
        layout = "bare LVAE (original / microSplit-reproducibility)"
        print(f"  dropped {len(dropped)} loss-side keys: {dropped}")
    print(f"Checkpoint {ckpt_path}: {len(state)} keys, layout = {layout}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [k for k in unexpected if not k.startswith("noise_model.")]
    # `model.parameter_net.*` aliases `model.output_layer.*` (chained assignment
    # in LadderVAE); pre-alias checkpoints saved only output_layer.*, which
    # populates the same tensors — a false-positive "missing" (bug catalog B2).
    missing = [
        k
        for k in missing
        if not k.startswith("noise_model.")
        and not k.startswith("model.parameter_net.")
    ]
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
    early_stop_patience: Optional[int] = None,
) -> Trainer:
    """Build a Lightning Trainer. `devices=1` preserves single-GPU behavior.

    Besides `last.ckpt`, the lowest-`val_loss` epoch is kept as `best.ckpt` --
    the equivalent of disentangle's `BaselineVAECL_best.ckpt`, which is what the
    published numbers were evaluated from. `early_stop_patience` mirrors
    disentangle's `training.earlystop_patience` (monitor `val_loss`); None = off.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    extra_callbacks: list[Any] = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )
    ]
    if early_stop_patience is not None:
        extra_callbacks.append(
            # Check after validation, never at train-epoch end: a checkpoint saved
            # after the last batch but before the epoch is marked complete resumes
            # into an empty epoch whose train-end hook has no val_loss yet (H24 job
            # 43359113, vault errors.md E3).
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=early_stop_patience,
                check_on_train_epoch_end=False,
            )
        )
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
                filename=f"ht_lif24_{DATASET_TAG}_{experiment_name}",
                save_last=True,
            ),
            *extra_callbacks,
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


def _save_prediction_bundle(
    save_dir: Path,
    arr_hwc: np.ndarray,
    gt_hwc: np.ndarray,
    inp: np.ndarray,
    frames: list[tuple[str, int, int]],
    data_root: Path,
    exposure: str,
    mmse_count: int,
    ckpt_path: Optional[str],
    seed: int,
    sets: Optional[list[str]] = None,
    eval_split: str = "val",
) -> None:
    """Write the same bundle as `microsplit_lif24_5ms_reference.py`: pred/gt/input
    TIFF stacks (SYXC / SYX, frame-aligned) plus index.json mapping each slot to
    (set, frame_idx) in the raw TIFFs, with per-frame range-invariant PSNR.

    The interior variant crops 16 px on each side — NOT because this pipeline has
    a dead stitching border (it does not), but to evaluate on the identical pixel
    support as the reference stack, whose stitcher never writes that border.
    """
    interior = 16
    n_frames, n_ch = arr_hwc.shape[0], arr_hwc.shape[-1]
    hi_y, hi_x = arr_hwc.shape[1] - interior, arr_hwc.shape[2] - interior

    per_frame = np.zeros((n_frames, n_ch))
    per_frame_int = np.zeros((n_frames, n_ch))
    for k in range(n_frames):
        for c in range(n_ch):
            per_frame[k, c] = _psnr(arr_hwc[k, ..., c], gt_hwc[k, ..., c])
            per_frame_int[k, c] = _psnr(
                arr_hwc[k, interior:hi_y, interior:hi_x, c],
                gt_hwc[k, interior:hi_y, interior:hi_x, c],
            )

    counts = _set_frame_counts(str(data_root / exposure), exposure)
    if len(frames) != n_frames:
        raise ValueError(
            f"bundle frame list has {len(frames)} entries but {n_frames} frames "
            "were predicted"
        )

    target_chs = CH_IDX_LIST[:-1]
    frames_meta = []
    for slot, (s, k, g) in enumerate(frames):
        src_5ms = str(data_root / exposure / s / f"uSplit_{exposure}.tif")
        src_hs = str(
            data_root
            / HIGHSNR_EXPOSURE_DURATION
            / s
            / f"uSplit_{HIGHSNR_EXPOSURE_DURATION}.tif"
        )
        frames_meta.append(
            {
                "slot": slot,
                "set": s,
                "frame_idx": k,
                "global_frame_idx": g,
                "src_5ms": src_5ms,
                "src_500ms": src_hs,
                "target_recipe": f"tifffile.imread(src_5ms)[{k}][{target_chs}]",
                "input_recipe": f"tifffile.imread(src_5ms)[{k}][{CH_IDX_LIST[-1]}]",
                "gt_recipe": f"tifffile.imread(src_500ms)[{k}][{target_chs}]",
                "rangeinvpsnr": [round(float(x), 4) for x in per_frame[slot]],
                "rangeinvpsnr_interior": [
                    round(float(x), 4) for x in per_frame_int[slot]
                ],
            }
        )

    save_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        save_dir / f"pred_{exposure}.tif", arr_hwc.astype(np.float32), bigtiff=True
    )
    tifffile.imwrite(
        save_dir / f"gt_{HIGHSNR_EXPOSURE_DURATION}.tif",
        gt_hwc.astype(np.float32),
        bigtiff=True,
    )
    tifffile.imwrite(
        save_dir / f"input_{exposure}.tif", inp.astype(np.float32), bigtiff=True
    )
    index = {
        "dataset": "HT_LIF24",
        "pipeline": "careamics-ng (scripts/microsplit_lif24_5ms.py)",
        "exposure": exposure,
        "highsnr_exposure": HIGHSNR_EXPOSURE_DURATION,
        "eval_split": eval_split,
        "sets": sets,
        "channel_idx_list": CH_IDX_LIST,
        "target_raw_channels": CH_IDX_LIST[:-1],
        "input_raw_channel": CH_IDX_LIST[-1],
        "saved_axes": "SYXC",
        "n_frames_saved": n_frames,
        "mmse_count": mmse_count,
        "interior_margin_px": interior,
        "interior_note": (
            "This pipeline stitches full frames (no dead border); the interior "
            "PSNR crops 16 px/side only to match the reference stack's support."
        ),
        "checkpoint": {
            "path": ckpt_path,
            "sha256": _sha256(Path(ckpt_path)) if ckpt_path else None,
        },
        "seed": seed,
        "set_frame_counts_5ms": counts,
        "files": {
            f"pred_{exposure}.tif": "MMSE predictions, float32 (S, Y, X, C)",
            f"gt_{HIGHSNR_EXPOSURE_DURATION}.tif": "high-SNR GT, float32 (S, Y, X, C)",
            f"input_{exposure}.tif": "raw superimposed input, float32 (S, Y, X)",
        },
        "frames": frames_meta,
    }
    (save_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"Saved prediction bundle ({n_frames} frames) to {save_dir}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    predict_sets: Optional[list[str]] = None,
    eval_split: str = "val",
    save_dir: Optional[Path] = None,
    mmse_count: int = 1,
    ckpt_path: Optional[str] = None,
    seed: int = 42,
) -> None:
    data_path = data_root / exposure
    if predict_sets:
        # All frames of the given Sets (matches the reference-pipeline bundle);
        # frame k of the outputs is frame k of the Sets' TIFF concatenation.
        va_in, _ = load_sets_arrays(
            str(data_path), exposure, CH_IDX_LIST, predict_sets
        )
    elif eval_split == "canonical-test":
        # The paper's held-out test split (10 frames) — same frames the original
        # `disentangle` evaluation uses, see scripts/microsplit_lif24_5ms_disentangle.py.
        _, _, _, _, va_in, _ = load_split_arrays(str(data_path), exposure, CH_IDX_LIST)
    else:
        _, _, va_in, _, _, _ = load_split_arrays(str(data_path), exposure, CH_IDX_LIST)
    pred_cfg = _predict_cfg(config, patch_size)
    dm.pred_config = pred_cfg
    dm.pred_input = [va_in]
    dm.setup("predict")
    predictions = trainer.predict(model, datamodule=dm)
    # `MicroSplitModule.predict_step` returns (prediction, uncertainty) per batch
    # since the predict-MMSE-std work; `convert_prediction` wants the predictions
    # alone. The uncertainty is the per-pixel spread across the MMSE draws and is
    # `None` when `--mmse-count 1`.
    predictions = [batch[0] for batch in predictions]
    stitched, _ = convert_prediction(predictions, tiled=True, restore_shape=True)
    arr = np.concatenate(stitched, axis=0) if len(stitched) > 1 else stitched[0]
    if arr.ndim == 4 and arr.shape[1] == OUTPUT_CHANNELS:
        arr = np.moveaxis(arr, 1, -1)
    print(f"Predictions shape: {arr.shape}")

    # Evaluation against 500ms high-SNR reference
    highsnr_path = data_root / HIGHSNR_EXPOSURE_DURATION
    if predict_sets:
        _, hs_target = load_sets_arrays(
            str(highsnr_path), HIGHSNR_EXPOSURE_DURATION, CH_IDX_LIST, predict_sets
        )
    elif eval_split == "canonical-test":
        _, _, _, _, _, hs_target = load_split_arrays(
            str(highsnr_path), HIGHSNR_EXPOSURE_DURATION, CH_IDX_LIST
        )
    else:
        _, _, _, hs_target, _, _ = load_split_arrays(
            str(highsnr_path), HIGHSNR_EXPOSURE_DURATION, CH_IDX_LIST
        )
    hs_target_hwc = np.moveaxis(hs_target, 1, -1)
    metrics = compute_stats([hs_target_hwc], [arr])
    Path(output_metrics).parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    frame_aligned = bool(predict_sets) or eval_split == "canonical-test"
    if frame_aligned:
        # support-identical comparison against the reference stack (16 px crop)
        m = 16
        hy, hx = arr.shape[1] - m, arr.shape[2] - m
        metrics_interior = compute_stats(
            [hs_target_hwc[:, m:hy, m:hx, :]], [arr[:, m:hy, m:hx, :]], verbose=False
        )
        print(f"Metrics (interior, {m}px crop): {metrics_interior}")
    print(f"Metrics: {metrics}")

    if save_dir is not None:
        frames = (
            _frames_for_sets(data_root, exposure, predict_sets)
            if predict_sets
            else _frames_for_split(data_root, exposure, eval_split)
        )
        _save_prediction_bundle(
            save_dir,
            arr,
            hs_target_hwc,
            va_in[:, 0],
            frames,
            data_root,
            exposure,
            mmse_count,
            ckpt_path,
            seed,
            sets=predict_sets,
            eval_split="sets" if predict_sets else eval_split,
        )

    saved = save_preview_images(arr, hs_target_hwc, preview_dir)
    print(f"Saved {len(saved)} preview PNGs to {preview_dir}")


def log_all_parameters(args) -> None:
    """Dump every CLI argument and every pinned constant, before anything runs."""
    bar = "=" * 70
    print(f"\n{bar}\nRESOLVED PARAMETERS\n{bar}")
    print("-- CLI arguments (after argparse defaults) --")
    for k in sorted(vars(args)):
        print(f"  {k:28s} = {getattr(args, k)!r}")
    print("-- pinned module constants (never on the CLI) --")
    for name in (
        "DEFAULT_DATA_ROOT", "EXPOSURE_DURATION", "HIGHSNR_EXPOSURE_DURATION", "DATASET_TAG",
        "CH_IDX_LIST", "Z_DIMS", "N_FILTERS", "MULTISCALE_COUNT",
        "OUTPUT_CHANNELS", "GRID_SIZE", "TRAINER_PRECISION",
        "TRAINER_GRADIENT_CLIP_VAL", "TRAINER_GRADIENT_CLIP_ALGORITHM",
        "NM_N_GAUSSIAN", "NM_N_COEFF", "NM_MIN_SIGMA", "NM_N_EPOCHS",
        "VAL_FRACTION", "TEST_FRACTION", "N_PREVIEW_SAMPLES",
    ):
        print(f"  {name:28s} = {globals()[name]!r}")
    print(f"{bar}\n")


def log_resolved_config(config: Any) -> None:
    """Dump the parts of the built configuration that decide what is optimised."""
    bar = "=" * 70
    algo = config.algorithm_config
    print(f"\n{bar}\nRESOLVED CONFIGURATION\n{bar}")
    print(f"  loss.gaussian_likelihood_weight    = "
          f"{algo.loss.gaussian_likelihood_weight}")
    print(f"  loss.noise_model_likelihood_weight = "
          f"{algo.loss.noise_model_likelihood_weight}")
    for attr in ("z_dims", "n_filters", "output_channels", "predict_logvar",
                 "multiscale_count", "encoder_conv_strides",
                 "decoder_conv_strides"):
        if hasattr(algo.model, attr):
            print(f"  model.{attr:28s} = {getattr(algo.model, attr)!r}")
    print(f"{bar}\n")


def print_noise_model_banner(model: MicroSplitModule, config: Any, nm_paths) -> None:

    """State unambiguously, in the log, whether this run uses a noise model."""
    raw = getattr(model, "_raw_noise_model", None)
    loss = config.algorithm_config.loss
    enabled = raw is not None and loss.noise_model_likelihood_weight > 0
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"NOISE MODEL: {'ENABLED' if enabled else 'DISABLED'}")
    print(f"{bar}")
    print(f"  noise_model_paths                     : {nm_paths}")
    print(f"  model._raw_noise_model                : {raw}")
    print(f"  n_gaussian / n_coeff / min_sigma      : "
          f"{[(m.n_gaussian, m.n_coeff, float(m.min_sigma.reshape(-1)[0])) for m in (getattr(raw, f'nmodel_{i}') for i in range(len(raw)))] if raw is not None else 'n/a -- no noise model'}")
    print(f"  loss.noise_model_likelihood_weight    : "
          f"{loss.noise_model_likelihood_weight}")
    print(f"  loss.gaussian_likelihood_weight       : "
          f"{loss.gaussian_likelihood_weight}")
    if enabled:
        print("  -> denoiSplit: the GMM noise-model likelihood IS part of the loss.")
    else:
        print("  -> pure muSplit: NO noise model anywhere in the loss. The only")
        print("     likelihood is the Gaussian one with a per-pixel predicted")
        print("     logvar. Nothing signal-dependent models the noise.")
    print(f"{bar}\n")


def main(args) -> None:
    configure_dataset(args.exposure, args.channel_idx_list)
    log_all_parameters(args)
    L.seed_everything(args.seed, workers=True)
    print(f"[BRANCH] seed_everything({args.seed}, workers=True)")
    data_root = Path(args.data_root)
    exposure = EXPOSURE_DURATION
    data_path = data_root / exposure
    noise_model_dir = (
        BASE_DIR / "noise_models" / f"{DATASET_TAG}_ngds_{args.experiment_name}"
    )
    output_dir = BASE_DIR / "lvae_checkpoints" / DATASET_TAG / args.experiment_name
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
        print("[BRANCH] noise model: --no-noise-model -> nm_paths=None, "
              "pure muSplit, NO N2V and NO GMM fit will run")
    elif args.noise_model_paths is not None:
        nm_paths = [Path(p) for p in args.noise_model_paths]
        print(f"[BRANCH] noise model: loading {len(nm_paths)} pretrained .npz -> "
              f"{[str(p) for p in nm_paths]}")
    elif args.skip_training:
        # Predict-only: NM is loss-side, not needed for forward pass
        nm_paths = None
        print("[BRANCH] noise model: --skip-training -> nm_paths=None "
              "(loss-side only, unused at predict)")
    else:
        print("[BRANCH] noise model: fitting from scratch -> N2V then GMM")
        # N2V + GMM fit is single-GPU (separate CAREamist Trainer inside).
        nm_input = prepare_noise_model_data(data_path, exposure)
        # The N2V pass is ~1 h and is stochastic. When sweeping GMM
        # hyperparameters every arm must see the SAME signal estimate, else the
        # N2V draw confounds the comparison -- so cache it and reuse.
        cache = Path(args.n2v_pred_cache) if args.n2v_pred_cache else None
        if cache is not None and cache.suffix != ".npy":
            raise ValueError(f"--n2v-pred-cache must end in .npy, got {cache}")
        if cache is not None and cache.exists():
            print(f"Loading cached N2V predictions from {cache}")
            n2v_pred = np.load(cache)
            if n2v_pred.shape != nm_input.shape:
                raise ValueError(
                    f"Cached N2V predictions {n2v_pred.shape} do not match the "
                    f"noise-model input {nm_input.shape}: {cache}"
                )
        else:
            n2v_pred = train_n2v(
                nm_input,
                args.experiment_name,
                noise_model_dir,
                n2v_num_epochs=args.n2v_num_epochs,
                patch_size=tuple(args.patch_size),
                batch_size=per_gpu_batch,
                seed=args.seed,
            )
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, n2v_pred)
                print(f"Cached N2V predictions to {cache}")
        nm_paths = fit_noise_models(
            n2v_pred,
            nm_input,
            noise_model_dir,
            n_gaussian=args.nm_n_gaussian,
            n_coeff=args.nm_n_coeff,
            min_sigma=args.nm_min_sigma,
            n_epochs=args.nm_n_epochs,
            global_signal_range=args.nm_global_signal_range,
        )

    if args.fit_noise_model_only:
        print(f"--fit-noise-model-only: stopping after the GMM fit. {nm_paths}")
        return

    # ------- optional N2V denoising of the input -------
    if not args.denoise_input:
        print("[BRANCH] --denoise-input NOT set -> network input stays the RAW "
              "noisy superimposed channel; no N2V anywhere in this run")
    if args.denoise_input:
        print("[BRANCH] --denoise-input SET -> N2V trained on the train split and "
              "applied to every frame, replacing the network input")
        if args.predict_sets:
            raise ValueError(
                "--denoise-input is only wired for the val / canonical-test "
                "splits; --predict-sets loads frames by Set and would misalign."
            )
        global _DENOISED_INPUT
        _DENOISED_INPUT = build_denoised_input(
            data_path,
            exposure,
            args.experiment_name,
            noise_model_dir,
            n2v_num_epochs=args.n2v_num_epochs,
            patch_size=tuple(args.patch_size),
            batch_size=per_gpu_batch,
            seed=args.seed,
            cache=(
                Path(args.denoise_input_cache) if args.denoise_input_cache else None
            ),
        )

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
        denoisplit_weight=args.denoisplit_weight,
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
    # PR #1053 / "rm nm from conf": the noise model and the MMSE sample count
    # left the configuration and are now injected on the module directly.
    if nm_paths:
        model.set_noise_model(nm_paths)
    log_resolved_config(config)
    print_noise_model_banner(model, config, nm_paths)
    model.n_samples = args.mmse_count[0]
    if args.compile:
        # Compiled in-place at `configure_model` time, so checkpoints stay
        # interchangeable with uncompiled runs. Under DDP this still gets
        # Dynamo's DDPOptimizer: the compiled forward runs inside DDP's own
        # forward, which is what activates the allreduce-overlap graph split.
        request_model_compilation(model, mode=args.compile_mode)
    trainer = create_trainer(
        config,
        output_dir,
        args.experiment_name,
        devices=args.devices,
        strategy=strategy,
        sync_batchnorm=sync_bn,
        early_stop_patience=args.early_stop_patience,
    )

    if args.skip_training:
        if args.pretrained_ckpt is None:
            raise ValueError("--skip-training requires --pretrained-ckpt")
        load_pretrained_model(model, args.pretrained_ckpt)
        dm.setup("fit")
    else:
        if args.pretrained_ckpt is not None:
            load_pretrained_model(model, args.pretrained_ckpt)
        # --resume-ckpt restores weights, optimizer, scheduler, epoch and the
        # callbacks' state (best val_loss, early-stop counter), so a run longer
        # than one SLURM wall can be chained. "auto" = this experiment's own
        # last.ckpt when it exists, else a fresh start -- the same command line
        # then serves as part 1 and part 2 of a chain.
        resume = args.resume_ckpt
        if resume == "auto":
            last = output_dir / "checkpoints" / "last.ckpt"
            resume = str(last) if last.exists() else None
        print(f"[BRANCH] resume_ckpt={resume!r}")
        trainer.fit(model, datamodule=dm, ckpt_path=resume)

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

    print(f"[BRANCH] skip_training={args.skip_training}  "
          f"skip_predict={args.skip_predict}  devices={args.devices}  "
          f"mmse_counts={args.mmse_count}")
    if not args.skip_predict:
        predict_trainer = (
            trainer
            if args.devices == 1
            else create_trainer(config, output_dir, args.experiment_name)
        )
        save_dir = None
        if args.predict_sets:
            save_dir = (
                Path(args.save_predictions_dir)
                if args.save_predictions_dir
                else BASE_DIR
                / "careamics_predictions"
                / f"{DATASET_TAG}_{'_'.join(args.predict_sets)}"
            )
        elif args.eval_split == "canonical-test":
            save_dir = (
                Path(args.save_predictions_dir)
                if args.save_predictions_dir
                else BASE_DIR / "careamics_predictions" / f"{DATASET_TAG}_canonical_test"
            )
        if save_dir is None:
            save_dir = (
                Path(args.save_predictions_dir)
                if args.save_predictions_dir
                else BASE_DIR
                / "careamics_predictions"
                / f"{DATASET_TAG}_{args.experiment_name}_{args.eval_split}"
            )
        for mc in args.mmse_count:
            print(f"\n########## PREDICT + EVAL at mmse_count={mc} ##########")
            model.n_samples = mc
            print_noise_model_banner(model, config, nm_paths)
            predict_and_eval(
                model,
                predict_trainer,
                config,
                dm,
                exposure,
                data_root,
                tuple(args.patch_size),
                str(Path(output_metrics).with_name(
                    f"{Path(output_metrics).stem}_mmse{mc}.json"
                )),
                preview_dir / f"mmse{mc}",
                predict_sets=args.predict_sets,
                eval_split=args.eval_split,
                save_dir=save_dir / f"mmse{mc}",
                mmse_count=mc,
                ckpt_path=args.pretrained_ckpt,
                seed=args.seed,
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
    parser.add_argument(
        "--exposure",
        type=str,
        default=EXPOSURE_DURATION,
        choices=["2ms", "3ms", "5ms", "20ms", "500ms"],
        help="Training/input exposure. Evaluation GT is always the 500ms capture.",
    )
    parser.add_argument(
        "--channel-idx-list",
        type=int,
        nargs="+",
        default=CH_IDX_LIST,
        help="Raw ND2 channels: targets..., then the real superimposed input. "
        "2-split [0 1 8], 3-split [1 2 3 17], 4-split [0 1 2 3 18].",
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
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="Stop when val_loss has not improved for this many epochs "
        "(disentangle training.earlystop_patience). Default: off.",
    )
    parser.add_argument(
        "--resume-ckpt",
        type=str,
        default=None,
        help="Resume training state from a Lightning checkpoint, or 'auto' for "
        "this experiment's own last.ckpt if present. Unlike --pretrained-ckpt it "
        "restores optimizer, scheduler, epoch and callback state.",
    )
    parser.add_argument("--n2v-num-epochs", type=int, default=10)
    parser.add_argument(
        "--mmse-count",
        type=int,
        nargs="+",
        default=[MMSE_COUNT],
        help="Number of posterior samples averaged at predict time (MMSE). "
        "Accepts several values -- predict + eval runs once per value, each "
        "writing its own metrics json, previews and prediction bundle.",
    )

    # torch.compile
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the LVAE with torch.compile. Adds a one-off warm-up cost "
        "on the first batch of each distinct input shape.",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode. Only used with --compile.",
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
    parser.add_argument(
        "--nm-n-gaussian",
        type=int,
        default=NM_N_GAUSSIAN,
        help=f"GMM components per channel. Default {NM_N_GAUSSIAN} (careamics "
        "default); the original MicroSplit noise models use 6.",
    )
    parser.add_argument(
        "--nm-n-coeff",
        type=int,
        default=NM_N_COEFF,
        help=f"Polynomial coefficients per GMM parameter. Default {NM_N_COEFF} "
        "(careamics default); the original MicroSplit noise models use 4.",
    )
    parser.add_argument(
        "--nm-min-sigma",
        type=float,
        default=NM_MIN_SIGMA,
        help=f"Lower clamp on the GMM VARIANCE (not sigma). Default "
        f"{NM_MIN_SIGMA} => sigma floor {NM_MIN_SIGMA ** 0.5:.2f} raw intensity "
        "units. The original MicroSplit noise models use 0.125. Where the clamp "
        "binds the variance branch gets no gradient and the GMM degenerates to a "
        "homoscedastic Gaussian -- see report_noise_model_diagnostics.",
    )
    parser.add_argument(
        "--nm-n-epochs",
        type=int,
        default=NM_N_EPOCHS,
        help=f"GMM fit steps (one 250k-pair minibatch each). Default {NM_N_EPOCHS}.",
    )
    parser.add_argument(
        "--nm-global-signal-range",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Initialise every channel's GMM from the signal extrema taken over "
        "ALL channels instead of per channel. False = careamics default.",
    )
    parser.add_argument(
        "--n2v-pred-cache",
        type=str,
        default=None,
        help="Path to a .npy of N2V predictions. Loaded if it exists, otherwise "
        "N2V runs and its output is written there. Use one cache across a GMM "
        "hyperparameter sweep so the signal estimate is held constant.",
    )
    parser.add_argument(
        "--denoise-input",
        action="store_true",
        help="Replace the noisy superimposed input with its N2V denoising "
        "(trained on the train split, applied to all frames). The alternative "
        "to a loss-side noise model: clean the input instead of modelling the "
        "noise. Combines with --no-noise-model or with a noise model.",
    )
    parser.add_argument(
        "--denoise-input-cache",
        type=str,
        default=None,
        help="Path to a .npy of the N2V-denoised input, (N, 1, Y, X). Loaded if "
        "it exists, otherwise written after the N2V pass. Share it between the "
        "with- and without-noise-model arms so both see the same input.",
    )
    parser.add_argument(
        "--fit-noise-model-only",
        action="store_true",
        help="Fit N2V + the GMMs, print diagnostics, and exit before the LVAE. "
        "Cheap screening arm of a noise-model sweep.",
    )
    parser.add_argument(
        "--denoisplit-weight",
        type=float,
        default=0.9,
        help="Weight of the noise-model likelihood in the loss; the Gaussian "
        "(muSplit) likelihood gets 1 - this. Default 0.9. Ignored with "
        "--no-noise-model.",
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
        "--predict-sets",
        nargs="+",
        type=str,
        default=None,
        help="Predict on ALL frames of these Sets (e.g. Set1) instead of the val "
        "split, and save a pred/gt/input TIFF bundle + index.json matching the "
        "reference-pipeline bundle (microsplit_lif24_5ms_reference.py). Metrics "
        "are then over these frames, NOT comparable to metrics_history.md rows.",
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        default="val",
        choices=["val", "canonical-test"],
        help="Frame set to predict + evaluate on when --predict-sets is not given. "
        "'val' (default) = the 12-frame validation split, comparable to most rows "
        "of metrics_history.md. 'canonical-test' = the paper's 10-frame held-out "
        "test split (Set1 f3,4,5,9,10,11,15,16,17 + Set2 f1), the same frames "
        "microsplit_lif24_5ms_disentangle.py --eval-split canonical-test uses, so "
        "the bundles are frame-aligned and directly comparable.",
    )
    parser.add_argument(
        "--save-predictions-dir",
        type=str,
        default=None,
        help="Bundle output dir for --predict-sets / --eval-split canonical-test. "
        "Default: scripts/careamics_predictions/<exposure>_<sets>/",
    )
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
    if args.predict_sets and args.eval_split != "val":
        parser.error("--predict-sets and --eval-split are mutually exclusive")
    main(args)
