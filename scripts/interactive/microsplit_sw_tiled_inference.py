# %% [markdown]
# # MicroSplit — sliding-window tiled inference
#
# Loads a pre-trained MicroSplit checkpoint, runs prediction with the dense
# `SlidingWindowTiledPatching` strategy (overlapping kept regions, uniform
# per-pixel MMSE coverage `K^d`), stitches by averaging via the `_TileAccumulator`
# helpers in `careamics.dev.sliding_window_tiled_pred`, and saves a single `.npz`
# keyed by input-image identifier.
#
# We can *not* use `convert_prediction(..., tiled=True)` here: its underlying
# `stitch_single_prediction` pastes by direct assignment, which would overwrite
# at the overlapping kept-region pixels instead of averaging them.

# %% imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from careamics.dev.sliding_window_tiled_pred import (
    compute_stride_for_mmse_count,
    sw_tiled_prediction,
)

from scripts.config_factory import pkl_load
from scripts.dataset_factory import build_pred_dataset
from scripts.io import npz_key, save_predictions_npz
from scripts.microsplit_factory import build_microsplit_module

# %% configuration (CL-args placeholder)
# TODO: lift to argparse / typer flags once the script is stable.
ROOT = Path("/project/careamics/switi")
DATA_ROOT = ROOT / "data"
CKPT_ROOT = ROOT / "ckpts"
OUT_ROOT = ROOT / "results"
DATASET = "PaviaATN"
SPLIT = "test"
OVERLAP = [32, 32]  # for 3D experiments switch to e.g. [0, 32, 32]
MMSE_COUNT = 2  # target effective per-pixel coverage (K^d); STRIDE derived below
STRIDE_Z: int | None = None  # 3D only; ignored for 2D experiments
BATCH_SIZE = 128
SAVE_DIR = OUT_ROOT / DATASET / "predictions" / "sw_inner_tiling"

data_dir = DATA_ROOT / DATASET
ckpt_path = CKPT_ROOT / DATASET / "BaselineVAECL_best.ckpt"
pkl_path = CKPT_ROOT / DATASET / "config.pkl"

# %% derive SW stride from the target MMSE count
# Given tile size (from pkl) and `OVERLAP`, pick the smallest per-pixel coverage
# >= `MMSE_COUNT`. YX stride is constrained symmetric; for 3D `STRIDE_Z` is
# explicit and the YX subproblem solves `K_y * K_x >= ceil(MMSE_COUNT / K_z)`.
_pkl_data = pkl_load(pkl_path)["data"]
_is_3d = bool(_pkl_data.get("mode_3D", False))
_img = int(_pkl_data["image_size"])
_patch_size = [_pkl_data["depth3D"], _img, _img] if _is_3d else [_img, _img]
STRIDE, _effective = compute_stride_for_mmse_count(
    _patch_size, OVERLAP, MMSE_COUNT,
    stride_z=STRIDE_Z if _is_3d else None,
)
print(
    f"patch_size={_patch_size}, overlap={OVERLAP}: "
    f"target MMSE count = {MMSE_COUNT}, effective = {_effective}, "
    f"stride = {STRIDE}"
)

# %% build prediction dataset
# Non-None stride routes `get_predict_config` to the SlidingWindowTiledPatchingConfig
# branch. Stats loaded from <data_dir>/stats.json or recomputed on first call.
dataset = build_pred_dataset(
    data_dir=data_dir,
    pkl_path=pkl_path,
    name=DATASET,
    split=SPLIT,
    overlap=OVERLAP,
    stride=STRIDE,
    batch_size=BATCH_SIZE,
)
print(f"dataset: n_patches={len(dataset)}, mode={dataset.config.mode}")

# %% build dataloader
loader = DataLoader(
    dataset,
    batch_size=dataset.config.batch_size,
    collate_fn=default_collate,
    num_workers=0,
    shuffle=False,
)

# %% debug — inspect one batch
_first = next(iter(loader))
print(
    "first batch:"
    f"\n  input shape   = {tuple(_first[0].data.shape)}"
    f"\n  region_spec   = {_first[0].region_spec}"
    f"\n  total_tiles   = {int(_first[0].region_spec['total_tiles'][0])}"
)

L = tuple(_first[0].data.shape)[1]
fig, axes = plt.subplots(4, L, figsize=(5*L, 20))
for i in range(4):
    for l in range(L):
        axes[i, l].imshow(_first[0].data[i, l], cmap="gray")
        axes[i, l].axis("off")
fig.tight_layout()

del _first

# %% build model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Per-tile model draws are hardcoded to 1: canonical for SW, per-pixel coverage
# comes from the patching geometry derived above.
model = build_microsplit_module(
    ckpt_path=ckpt_path,
    pkl_path=pkl_path,
    mmse_count=1,
    device=device,
)

# %% wire target stats from the dataset into the model
model.set_target_stats(
    dataset.normalization.target_means,
    dataset.normalization.target_stds,
)

# %% prediction loop — streaming SW averaging
preds_list, sources = sw_tiled_prediction(model, loader)

# %% build results dict
# `sources` is empty when all inputs are in-memory arrays (InMemoryImageStack
# uses the literal "array" sentinel). For path-backed inputs each entry is the
# source path string, so `npz_key` picks the filename stem.
results: dict[str, NDArray] = {}
for data_idx, pred in enumerate(preds_list):
    source = sources[data_idx] if sources else "array"
    results[npz_key(source, data_idx)] = pred
print(f"stitched {len(results)} image(s)")

# %% save
out_path = save_predictions_npz(results, SAVE_DIR)
print(f"wrote {len(results)} prediction(s) to {out_path}")

# %% quick visualization
first_key = next(iter(results))
first_pred = results[first_key]  # (S, output_channels, [Z], Y, X)
sample = first_pred[0]
if sample.ndim == 4:  # (C, Z, Y, X) -> pick mid Z slice
    sample = sample[:, sample.shape[1] // 2]
n_out = sample.shape[0]
fig, axes = plt.subplots(1, n_out, figsize=(4 * n_out, 4))
axes = np.atleast_1d(axes)
for c, ax in enumerate(axes):
    ax.imshow(sample[c], cmap="gray")
    ax.set_title(f"{first_key} — channel {c}")
    ax.axis("off")
fig.tight_layout()
plt.show()

# %%
