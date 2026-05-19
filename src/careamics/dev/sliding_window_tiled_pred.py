"""Sliding-window tiled prediction for posterior models (v1, MicroSplit-only).

Implements a dense-overlap tile stitcher that keeps *all* predicted pixels and
averages them across overlapping tiles, rather than the standard inner-tiling
approach of cropping each tile to its central region.

See ``swin_tiled_pred.md`` for design notes.
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


def _tile_paste_slices(spec: TileSpecs) -> tuple[Any, ...]:
    """Build the destination slice into an SC(Z)YX accumulator for a tile.

    Uses the **full** tile extent (``coords`` / ``patch_size``); the
    ``crop_coords``/``stitch_coords`` fields used by the legacy inner-tile
    stitcher are intentionally ignored here.
    """
    coords = spec["coords"]
    size = spec["patch_size"]
    sample_idx = int(spec["sample_idx"])
    spatial = tuple(
        slice(int(s), int(s) + int(length))
        for s, length in zip(coords, size, strict=True)
    )
    return (sample_idx, ..., *spatial)


def _paste_tile(acc: _TileAccumulator, tile: ImageRegionData) -> None:
    """Add a tile's data into the accumulator and bump the count."""
    spec: TileSpecs = tile.region_spec  # type: ignore[assignment]
    slices = _tile_paste_slices(spec)
    acc.sum[slices] += np.asarray(tile.data, dtype=np.float32)
    acc.count[slices] += 1.0
    acc.seen += 1


def _handle_border_tiles(acc: _TileAccumulator) -> None:
    """Placeholder for the border-MMSE correction.

    The first/last P×P panels of the image are not covered by enough sliding
    windows to reach MMSE-fixed status. The intended fix is to run a standalone
    MMSE prediction at the borders and weight-paste pixels P, P-1, ... towards
    the interior. Not implemented in v1.
    """
    # TODO: run a standalone MMSE prediction at the borders and weight-paste
    # P, P-1, ... towards the interior.
    return


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
            "Image data_idx=%d has %d uncovered pixel(s) — likely the border "
            "edge case (not handled in v1). Those pixels will be written as 0.",
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
    """Run dense-overlap sliding-window tiled prediction.

    Iterates the prediction dataloader once. For each batch the model's
    ``predict_step`` produces an already-denormalised MMSE mean per tile; tiles
    are pasted (summed) into a per-image accumulator at their full ``coords`` +
    ``patch_size`` location and a parallel count array tracks coverage. When all
    tiles for an image have arrived (per ``TileSpecs.total_tiles``), the sum is
    divided by the count and written to disk as a TIFF, and the accumulator is
    flushed.

    Parameters
    ----------
    model : MicroSplitModule
        Trained MicroSplit module. Caller is responsible for loading weights
        (e.g. via ``load_microsplit_from_checkpoint``).
    use_logger : bool
        Currently unused; kept for API compatibility with the handout.
    data_config : DataConfig or dict
        Configuration for the prediction ``CareamicsDataModule``. Dense overlap
        is driven by setting a small grid stride on the tiled-patching strategy
        in this config — no new patching strategy is introduced.
    inputs : sequence of NDArray or Path
        Prediction inputs. If items are paths, each output is saved as
        ``{stem}_pred.tif``; otherwise outputs are saved as
        ``pred_{data_idx:04d}.tif``.
    save_dir : Path or str
        Directory where TIFFs are written. Created if missing.
    """
    del use_logger  # TODO: wire up if/when needed

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
                    _handle_border_tiles(acc)
                    _finalize_and_save(
                        accumulators.pop(data_idx), save_dir, data_idx
                    )

    if accumulators:
        raise RuntimeError(
            "Prediction ended with incomplete images "
            f"(data_idx={sorted(accumulators)}). This indicates a mismatch "
            "between received and expected tile counts (TileSpecs.total_tiles)."
        )
