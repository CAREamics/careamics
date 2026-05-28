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

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from careamics.dev.sliding_window_tiled_pred import (
    _TileAccumulator,
    _allocate_accumulator,
    _move_input_to_device,
    _paste_tile,
    effective_mmse_count,
)
from careamics.lightning.prediction.convert_prediction import (
    decollate_image_region_data,
)

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
STRIDE = [4, 4]  # for 3D experiments switch to e.g. [1, 4, 4]
MMSE_COUNT = 1  # canonical for SW: per-pixel count comes from geometry
SAVE_DIR = Path("./predictions") / DATASET / "sw_tiled"

data_dir = DATA_ROOT / DATASET
ckpt_path = CKPT_ROOT / DATASET / "BaselineVAECL_best.ckpt"
pkl_path = CKPT_ROOT / DATASET / "config.pkl"

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
)
print(f"dataset: n_patches={len(dataset)}, mode={dataset.config.mode}")

# Print the geometric per-pixel sample count so the comparison vs the classical
# script is explicit. `K^d` where K = (patch - overlap) // stride per axis.
_patch_size = list(dataset.config.patching.patch_size)
_overlaps = list(dataset.config.patching.overlaps)
_stride = list(dataset.config.patching.stride)
_per_axis_K = [
    effective_mmse_count(p, s, o)
    for p, s, o in zip(_patch_size, _stride, _overlaps, strict=True)
]
print(
    f"per-axis K = {_per_axis_K}, "
    f"per-pixel sample count = {int(np.prod(_per_axis_K))}"
)

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
model.set_target_stats(
    dataset.normalization.target_means,
    dataset.normalization.target_stds,
)

# %% debug — inspect one batch
_first = next(iter(loader))
print(
    "first batch:"
    f"\n  input shape   = {tuple(_first[0].data.shape)}"
    f"\n  region_spec   = {_first[0].region_spec}"
    f"\n  total_tiles   = {int(_first[0].region_spec['total_tiles'][0])}"
)
del _first

# %% prediction loop — streaming SW averaging
# Per-image accumulators of running sum + count. Each predicted tile is cropped
# to its kept inner region (via crop_coords / crop_size from TileSpecs) and
# *added* (not assigned) into the accumulator at stitch_coords; per-pixel count
# tracks how many tiles contributed. Once an image's tile count reaches its
# reported total_tiles, the mean is computed and stored, freeing the accumulator.
output_channels = int(model.algorithm_config.model.output_channels)
accumulators: dict[int, _TileAccumulator] = {}
results: dict[str, NDArray] = {}

with torch.inference_mode():
    for batch_idx, batch in enumerate(loader):
        batch = _move_input_to_device(batch, device)
        mean_region_batch, _std = model.predict_step(batch, batch_idx)

        for tile in decollate_image_region_data(mean_region_batch):
            data_idx = int(tile.region_spec["data_idx"])
            acc = accumulators.get(data_idx)
            if acc is None:
                acc = _allocate_accumulator(tile, output_channels)
                accumulators[data_idx] = acc
            _paste_tile(acc, tile)

            if acc.is_complete():
                mean = np.divide(
                    acc.sum,
                    acc.count,
                    out=np.zeros_like(acc.sum),
                    where=acc.count > 0,
                )
                results[npz_key(acc.source, data_idx)] = mean
                del accumulators[data_idx]

if accumulators:
    raise RuntimeError(
        "Prediction ended with incomplete images "
        f"(data_idx={sorted(accumulators)}). This indicates a mismatch "
        "between received and expected tile counts (TileSpecs.total_tiles)."
    )

print(f"stitched {len(results)} image(s)")

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
