"""Sliding-window inner-tiled prediction for posterior models (MicroSplit-only).

Implements a dense-overlap inner-tile stitcher: each predicted tile is cropped
to its kept inner region (drop margin of ``overlap // 2`` per side, asymmetric
at image edges) and pasted at its ``stitch_coords`` with `+=` into a running
sum, with a parallel count array tracking coverage. Tile geometry is produced
by ``SlidingWindowTiledPatching``; effective per-pixel MMSE count is determined
by ``effective_mmse_count(patch_size, stride, overlap)`` and edge replication
in the patching strategy equalises border coverage with the interior.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch
from numpy.typing import NDArray

from careamics.config import VAEBasedAlgorithm
from careamics.config.data.data_config import DataConfig
from careamics.dataset.image_region_data import ImageRegionData
from careamics.dataset.patching import TileSpecs
from careamics.lightning.data.data_module import CareamicsDataModule
from careamics.lightning.modules.microsplit_module import MicroSplitModule
from careamics.lightning.prediction.convert_prediction import (
    decollate_image_region_data,
)
from careamics.utils import get_logger

logger = get_logger(__name__)


def load_microsplit_from_checkpoint(
    ckpt_path: Path | str,
    algorithm_config: VAEBasedAlgorithm,
) -> MicroSplitModule:
    """Instantiate a ``MicroSplitModule`` and load weights from a checkpoint.

    ``MicroSplitModule.__init__`` does not call ``save_hyperparameters()``, so the
    algorithm config must be supplied explicitly.

    Parameters
    ----------
    ckpt_path : pathlib.Path or str
        Path to the Lightning checkpoint file.
    algorithm_config : VAEBasedAlgorithm
        Algorithm configuration required to instantiate the module.

    Returns
    -------
    MicroSplitModule
        Module with weights loaded from the checkpoint.
    """
    return MicroSplitModule.load_from_checkpoint(
        checkpoint_path=str(ckpt_path),
        algorithm_config=algorithm_config,
        strict=True,
    )


def effective_mmse_count(patch_size: int, stride: int, overlap: int) -> int:
    """Per-axis effective MMSE count for `SlidingWindowTiledPatching`.

    Each pixel along the axis is covered by this many independent tile
    predictions (assuming `mmse_count = 1` in the model — each forward pass
    yields one stochastic draw). For a multi-axis pixel, the effective count
    is the product of this value across axes.

    Parameters
    ----------
    patch_size : int
        Tile size along the axis.
    stride : int
        Tile stride along the axis.
    overlap : int
        Overlap dropped from each adjacent tile pair (= 2 * margin per side).

    Returns
    -------
    int
        `max(1, (patch_size - overlap) // stride)`.
    """
    return max(1, (patch_size - overlap) // stride)


@dataclass
class _TileAccumulator:
    """Per-image accumulator for sliding-window tile averaging.

    ``sum`` and ``count`` share the same SC(Z)YX shape as the full image; each
    tile is added in-place at its ``(sample_idx, ..., *spatial_slice)`` location.
    """

    sum: NDArray
    count: NDArray
    expected_tiles: int
    seen: int
    source: str
    axes: str
    original_data_shape: tuple[int, ...]

    def is_complete(self) -> bool:
        """Whether all tiles for this image have been accumulated."""
        return self.seen >= self.expected_tiles


def _allocate_accumulator(tile: ImageRegionData) -> _TileAccumulator:
    """Allocate an accumulator sized after the full image carried by ``tile``."""
    spec: TileSpecs = tile.region_spec  # type: ignore[assignment]
    shape = tuple(int(d) for d in tile.data_shape)
    return _TileAccumulator(
        sum=np.zeros(shape, dtype=np.float32),
        count=np.zeros(shape, dtype=np.float32),
        expected_tiles=int(spec["total_tiles"]),
        seen=0,
        source=tile.source,
        axes=tile.axes,
        original_data_shape=tuple(int(d) for d in tile.original_data_shape),
    )


def _tile_paste_slices(
    spec: TileSpecs,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Build the source-crop and destination-stitch slices for a tile.

    Source slice indexes into the tile data (``C(Z)YX``) to extract its kept
    inner region (``crop_coords`` / ``crop_size``). Destination slice indexes
    into the ``SC(Z)YX`` accumulator at the tile's ``stitch_coords``.
    """
    crop_coords = spec["crop_coords"]
    crop_size = spec["crop_size"]
    stitch_coords = spec["stitch_coords"]
    sample_idx = int(spec["sample_idx"])
    source = (
        ...,
        *[
            slice(int(c), int(c) + int(sz))
            for c, sz in zip(crop_coords, crop_size, strict=True)
        ],
    )
    dest = (
        sample_idx,
        ...,
        *[
            slice(int(s), int(s) + int(sz))
            for s, sz in zip(stitch_coords, crop_size, strict=True)
        ],
    )
    return source, dest


def _paste_tile(acc: _TileAccumulator, tile: ImageRegionData) -> None:
    """Add a tile's cropped inner region into the accumulator and bump the count."""
    spec: TileSpecs = tile.region_spec  # type: ignore[assignment]
    source_slice, dest_slice = _tile_paste_slices(spec)
    cropped = np.asarray(tile.data, dtype=np.float32)[source_slice]
    acc.sum[dest_slice] += cropped
    acc.count[dest_slice] += 1.0
    acc.seen += 1


def _resolve_output_path(save_dir: Path, data_idx: int, source: str) -> Path:
    """Pick the output TIFF path for an image."""
    if source == "array":
        return save_dir / f"pred_{data_idx:04d}.tif"
    return save_dir / f"{Path(source).stem}_pred.tif"


def _finalize_and_save(
    acc: _TileAccumulator,
    save_dir: Path,
    data_idx: int,
) -> None:
    """Average sum by count and write the result as a TIFF."""
    if (acc.count == 0).any():
        n_uncovered = int((acc.count == 0).sum())
        logger.warning(
            "Image data_idx=%d has %d uncovered pixel(s). With "
            "SlidingWindowTiledPatching this should not happen — check your "
            "stride/overlap configuration. Those pixels will be written as 0.",
            data_idx,
            n_uncovered,
        )
    mean = np.divide(
        acc.sum,
        acc.count,
        out=np.zeros_like(acc.sum),
        where=acc.count > 0,
    )
    out_path = _resolve_output_path(save_dir, data_idx, acc.source)
    tifffile.imwrite(out_path, mean.astype(np.float32))
    logger.info("Wrote prediction for data_idx=%d to %s", data_idx, out_path)


def _move_input_to_device(
    batch: tuple[ImageRegionData, ...], device: torch.device
) -> tuple[ImageRegionData, ...]:
    """Return ``batch`` with ``batch[0].data`` moved to ``device``.

    ``ImageRegionData`` is a ``NamedTuple``; ``_replace`` is used to swap the
    ``data`` field without mutating the original.
    """
    input_region = batch[0]
    moved = input_region._replace(data=input_region.data.to(device))
    return (moved, *batch[1:])


def sw_tiled_prediction(
    model: MicroSplitModule,
    use_logger: bool,
    data_config: DataConfig | dict[str, Any],
    inputs: Sequence[NDArray | Path | str],
    save_dir: Path | str,
) -> None:
    """Run dense-overlap sliding-window inner-tiled prediction.

    Iterates the prediction dataloader once. For each batch the model's
    ``predict_step`` produces an already-denormalised MMSE mean per tile; each
    tile's kept inner region (per its ``crop_coords`` / ``crop_size``) is
    pasted at ``stitch_coords`` into a per-image accumulator with ``+=``, and
    a parallel count array tracks coverage. When all tiles for an image have
    arrived (per ``TileSpecs.total_tiles``), the sum is divided by the count
    and written to disk as a TIFF, and the accumulator is flushed.

    The effective per-pixel MMSE count is determined by the patching strategy:
    use ``SlidingWindowTiledPatchingConfig`` and the helper
    ``effective_mmse_count(patch_size, stride, overlap)`` to predict it. With
    ``edge_replication=True`` the border matches the interior. Set
    ``model.algorithm_config.mmse_count = 1`` for canonical behaviour; values
    > 1 multiply the effective count.

    Parameters
    ----------
    model : MicroSplitModule
        Trained MicroSplit module. Caller is responsible for loading weights
        (e.g. via ``load_microsplit_from_checkpoint``) and for setting
        ``model.algorithm_config.mmse_count = 1`` before invocation.
    data_config : DataConfig or dict
        Configuration for the prediction ``CareamicsDataModule``. Expected to
        carry a ``SlidingWindowTiledPatchingConfig`` as its patching strategy.
    inputs : sequence of NDArray or Path
        Prediction inputs. If items are paths, each output is saved as
        ``{stem}_pred.tif``; otherwise outputs are saved as
        ``pred_{data_idx:04d}.tif``.
    save_dir : Path or str
        Directory where TIFFs are written. Created if missing.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dm = CareamicsDataModule(data_config, pred_data=list(inputs))
    dm.setup("predict")
    loader = dm.predict_dataloader()

    accumulators: dict[int, _TileAccumulator] = {}

    model.eval()
    device = next(model.parameters()).device
    # TODO: revisit std handling — requires per-MMSE-sample exposure from
    # predict_step. v1 discards std_region_batch.
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            batch = _move_input_to_device(batch, device)
            mean_region_batch, _std_region_batch = model.predict_step(
                batch, batch_idx
            )
            tiles = decollate_image_region_data(mean_region_batch)

            for tile in tiles:
                data_idx = int(tile.region_spec["data_idx"])
                acc = accumulators.get(data_idx)
                if acc is None:
                    acc = _allocate_accumulator(tile)
                    accumulators[data_idx] = acc
                _paste_tile(acc, tile)

                if acc.is_complete():
                    _finalize_and_save(
                        accumulators.pop(data_idx), save_dir, data_idx
                    )

    if accumulators:
        raise RuntimeError(
            "Prediction ended with incomplete images "
            f"(data_idx={sorted(accumulators)}). This indicates a mismatch "
            "between received and expected tile counts (TileSpecs.total_tiles)."
        )
