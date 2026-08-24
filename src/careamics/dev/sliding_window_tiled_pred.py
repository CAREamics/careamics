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
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.optimize import brute
from torch.utils.data import DataLoader
from tqdm import tqdm

from careamics.dataset.image_region_data import ImageRegionData
from careamics.dataset.patching import TileSpecs
from careamics.lightning.modules.microsplit_module import MicroSplitModule
from careamics.lightning.prediction.convert_prediction import (
    decollate_image_region_data,
)
from careamics.utils import get_logger

logger = get_logger(__name__)


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
    return int(np.ceil((patch_size - overlap) / stride))


def compute_stride_for_mmse_count(
    patch_size: Sequence[int],
    overlap: Sequence[int],
    target_mmse_count: int,
    *,
    stride_z: int | None = None,
) -> tuple[tuple[int, int] | tuple[int, int, int], int]:
    # NOTE: should already be validated in config
    if patch_size[-1] != patch_size[-2]:
        raise ValueError(
            f"Only patches square in the XY dimensions are valid, got {patch_size}."
        )
    if overlap[-2] != overlap[-1]:  # TODO: not sure if validated in config
        raise ValueError(f"Overlaps must be equal in XY, got {overlap}.")

    crop_size_xy = patch_size[-2] - overlap[-1]  # equal for X and Y
    stride: tuple[int, int] | tuple[int, int, int]
    match patch_size, stride_z:
        case (_, _), _:
            if stride_z is not None:
                logger.warning(
                    "Ignoring parameter `stride_z` for 2D data in SWITi stride "
                    "calculation."
                )
            stride_xy = int(np.ceil(crop_size_xy / np.sqrt(target_mmse_count)))
            stride = (stride_xy, stride_xy)
        case (_, _, _), _ if stride_z is not None:
            crop_size_z = patch_size[0] - overlap[0]
            coverage_z = int(np.ceil(crop_size_z / stride_z))
            coverage_remaining = target_mmse_count / coverage_z
            stride_xy = int(np.ceil(crop_size_xy / np.sqrt(coverage_remaining)))
            stride = (stride_z, stride_xy, stride_xy)
        case (_, _, _), None:
            crop_size_z = patch_size[0] - overlap[0]
            stride_z, stride_xy = _search_3D_strides(
                crop_size_xy, crop_size_z, target_mmse_count
            )
            stride = (stride_z, stride_xy, stride_xy)
        case _:
            raise ValueError(
                f"Invalid `patch_size`, only 2D or 3D is allowed, got {patch_size}."
            )

    achieved_mmse_count = np.prod(
        [
            effective_mmse_count(ps, s, o)
            for ps, s, o in zip(patch_size, stride, overlap, strict=True)
        ]
    ).item()
    return stride, achieved_mmse_count


def _search_3D_strides(
    crop_size_xy: int, crop_size_z: int, target_mmse_count: int
) -> tuple[int, int]:

    def objective(value: tuple[int, int]):
        stride_xy, stride_z = value
        coverage_xy = int(np.ceil(crop_size_xy / stride_xy))
        coverage_z = int(np.ceil(crop_size_z / stride_z))

        primary_objective = coverage_xy**2 * coverage_z - target_mmse_count

        # must evaluate to greater than the target count
        if primary_objective < 0:
            return np.inf

        # secondary objective:
        # in the case of a tie, choose solution that minimizes abs(stride_z - stride_xy)
        secondary_objective = abs(stride_z - stride_xy)

        # Larger than the maximum possible abs(stride_z - stride_xy)
        tie_break_scale = max(crop_size_xy, crop_size_z) + 1

        return primary_objective * tie_break_scale + secondary_objective

    result = brute(
        objective,
        ranges=(
            # stride cannot be greater than half the crop size,
            # as it would result in uneven coverage
            slice(1, max(crop_size_xy // 2, 1) + 1),
            slice(1, max(crop_size_z // 2, 1) + 1),
        ),
        finish=None,
    )

    stride_xy, stride_z = result.astype(int).tolist()
    return stride_z, stride_xy


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


def _allocate_accumulator(
    tile: ImageRegionData, output_channels: int
) -> _TileAccumulator:
    """Allocate an accumulator sized after the full image carried by ``tile``.

    The spatial extent comes from ``tile.data_shape`` (the input image's
    ``SC(Z)YX``), while the channel dimension is overridden with
    ``output_channels`` so the buffer matches the model output (which may have a
    different number of channels than the input — e.g. MicroSplit unmixing).

    Parameters
    ----------
    tile : ImageRegionData
        A tile carrying input-image metadata and a ``TileSpecs`` ``region_spec``.
    output_channels : int
        Number of channels produced by the model (used as the buffer's C axis).

    Returns
    -------
    _TileAccumulator
        Zero-initialised sum/count accumulator sized to the full output image.
    """
    spec: TileSpecs = tile.region_spec  # type: ignore[assignment]
    input_shape = tuple(int(d) for d in tile.data_shape)
    shape = (input_shape[0], int(output_channels), *input_shape[2:])
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


def _finalize(acc: _TileAccumulator, data_idx: int) -> NDArray:
    """Average sum by count and return the mean array."""
    if (acc.count == 0).any():
        n_uncovered = int((acc.count == 0).sum())
        logger.warning(
            "Image data_idx=%d has %d uncovered pixel(s). With "
            "SlidingWindowTiledPatching this should not happen — check your "
            "stride/overlap configuration. Those pixels will be returned as 0.",
            data_idx,
            n_uncovered,
        )
    mean = np.divide(
        acc.sum,
        acc.count,
        out=np.zeros_like(acc.sum),
        where=acc.count > 0,
    )
    return mean.astype(np.float32)


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
    dataloader: DataLoader,
) -> tuple[list[NDArray], list[str]]:
    """Run dense-overlap sliding-window inner-tiled prediction.

    Iterates ``dataloader`` once. For each batch the model's ``predict_step``
    produces an already-denormalised MMSE mean per tile; each tile's kept inner
    region (per its ``crop_coords`` / ``crop_size``) is pasted at
    ``stitch_coords`` into a per-image accumulator with ``+=``, and a parallel
    count array tracks coverage. When all tiles for an image have arrived (per
    ``TileSpecs.total_tiles``), the sum is divided by the count and stored;
    after the loop the per-image means are returned sorted by ``data_idx``.

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
        and for setting ``model.algorithm_config.mmse_count = 1`` before invocation.
    dataloader : DataLoader
        Prediction dataloader. Expected to be built from a data module whose
        patching strategy is ``SlidingWindowTiledPatchingConfig``.

    Returns
    -------
    list of numpy.ndarray
        Per-image stitched predictions with axes ``SC(Z)YX``, sorted by
        ``data_idx``.
    list of str
        Per-image sources, in the same order. Empty if all sources equal
        ``"array"`` (mirroring ``convert_prediction``).
    """
    # Model output channel count drives the stitch buffer's C axis.
    # For MicroSplit this differs from the input image's channel count.
    output_channels = int(model.algorithm_config.model.output_channels)

    accumulators: dict[int, _TileAccumulator] = {}
    finalized: dict[int, tuple[NDArray, str]] = {}

    model.eval()
    device = next(model.parameters()).device
    # TODO: revisit std handling — requires per-MMSE-sample exposure from
    # predict_step. v1 discards std_region_batch.
    with torch.inference_mode():
        for batch_idx, batch in enumerate(
            tqdm(dataloader, total=len(dataloader), desc="Predicting")
        ):
            batch = _move_input_to_device(batch, device)
            mean_region_batch, _std_region_batch = model.predict_step(batch, batch_idx)
            tiles = decollate_image_region_data(mean_region_batch)

            for tile in tiles:
                data_idx = int(tile.region_spec["data_idx"])
                acc = accumulators.get(data_idx)
                if acc is None:
                    acc = _allocate_accumulator(tile, output_channels)
                    accumulators[data_idx] = acc
                _paste_tile(acc, tile)

                if acc.is_complete():
                    completed = accumulators.pop(data_idx)
                    finalized[data_idx] = (
                        _finalize(completed, data_idx),
                        completed.source,
                    )

    if accumulators:
        raise RuntimeError(
            "Prediction ended with incomplete images "
            f"(data_idx={sorted(accumulators)}). This indicates a mismatch "
            "between received and expected tile counts (TileSpecs.total_tiles)."
        )

    # TODO: directly write predictions on disk once debugging is finished
    predictions_output: list[NDArray] = []
    sources: list[str] = []
    for data_idx in sorted(finalized.keys()):
        arr, src = finalized[data_idx]
        predictions_output.append(arr)
        sources.append(src)

    if set(sources) == {"array"}:
        sources = []

    return predictions_output, sources
