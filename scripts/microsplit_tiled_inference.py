# %% [markdown]
# # MicroSplit — classical tiled inference
#
# Loads a pre-trained MicroSplit checkpoint, runs prediction with classical inner
# tiling (`TiledPatching`, non-overlapping kept regions), stitches via the canonical
# `convert_prediction(..., tiled=True)` path (same code path that powers
# `PredictionWriterCallback` + `TileWriteStrategy` for N2V/CARE), and saves a single
# `.npz` keyed by input-image identifier.
#
# Each cell is delimited by `# %%` markers — runnable in VSCode / PyCharm
# interactive mode or convertible to a notebook with `jupytext --to ipynb`.

# %% imports
from collections import defaultdict  # noqa: F401  (placeholder for inspection cells)
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from careamics.dataset.image_region_data import ImageRegionData  # noqa: F401
from careamics.dev.sliding_window_tiled_pred import _move_input_to_device
from careamics.lightning.prediction.convert_prediction import convert_prediction

from scripts.dataset_factory import build_pred_dataset
from scripts.io import npz_key, save_predictions_npz
from scripts.microsplit_factory import build_microsplit_module

# %% configuration (CL-args placeholder)
# TODO: lift to argparse / typer flags once the script is stable.
DATA_ROOT = Path("/project/careamics/switi/data")
CKPT_ROOT = Path("/project/careamics/switi/ckpts")
DATASET = "HT_LIF24_5ms"
SPLIT = "test"
OVERLAP = [32, 32]  # for 3D experiments switch to e.g. [0, 32, 32]
MMSE_COUNT = 50
SAVE_DIR = Path("./predictions") / DATASET / "tiled"

data_dir = DATA_ROOT / DATASET
ckpt_path = CKPT_ROOT / DATASET / "BaselineVAECL_best.ckpt"
pkl_path = CKPT_ROOT / DATASET / "config.pkl"

# %% build prediction dataset
# NOTE: Classical inner tiling: stride=None inside `get_predict_config` picks the
# TiledPatchingConfig branch. Stats are loaded from <data_dir>/stats.json or
# recomputed and cached on first call.
dataset = build_pred_dataset(
    data_dir=data_dir,
    pkl_path=pkl_path,
    name=DATASET,
    split=SPLIT,
    overlap=OVERLAP,
    stride=None,
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

# %% build model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_microsplit_module(
    ckpt_path=ckpt_path,
    pkl_path=pkl_path,
    mmse_count=MMSE_COUNT,
    device=device,
)

# %% wire target stats from the dataset into the model
# `predict_step` reads these to denormalize into target space.
model.set_target_stats(
    dataset.normalization.target_means,
    dataset.normalization.target_stds,
)

# %% debug — inspect one batch before launching the full loop
_first = next(iter(loader))
print(
    "first batch:"
    f"\n  input shape   = {tuple(_first[0].data.shape)}"
    f"\n  region_spec   = {_first[0].region_spec}"
    f"\n  total_tiles   = {int(_first[0].region_spec['total_tiles'][0])}"
)
del _first

# %% prediction loop
# Collect all batched mean regions; `convert_prediction(tiled=True)` then
# decollates, groups by `data_idx`, and stitches each image in one shot via
# `stitch_single_prediction` (direct paste — correct for non-overlapping kept
# regions).
predictions: list[ImageRegionData] = []
with torch.inference_mode():
    for batch_idx, batch in enumerate(loader):
        batch = _move_input_to_device(batch, device)
        mean_region_batch, _std = model.predict_step(batch, batch_idx)
        predictions.append(mean_region_batch)

preds_list, sources = convert_prediction(
    predictions, tiled=True, restore_shape=False
)
print(f"stitched {len(preds_list)} image(s)")

# %% build results dict
# `sources` is empty when all inputs are in-memory arrays (InMemoryImageStack
# uses the literal "array" sentinel). For path-backed inputs each entry is the
# source path string, so `npz_key` picks the filename stem.
results: dict[str, NDArray] = {}
for data_idx, pred in enumerate(preds_list):
    source = sources[data_idx] if sources else "array"
    results[npz_key(source, data_idx)] = pred

# %% save
out_path = save_predictions_npz(results, SAVE_DIR)
print(f"wrote {len(results)} prediction(s) to {out_path}")

# %% quick visualization
import matplotlib.pyplot as plt  # noqa: E402

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
