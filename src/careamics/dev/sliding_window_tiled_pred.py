"""Sliding-window inner-tiled (SWITi) prediction.

Functions for stitching densely sampled overlapping tiles produced by the
`SWITiPatching` strategy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import EllipsisType

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
    """Calculate per-axis effective MMSE count for `SwitiPatching`.

    The effective MMSE count for multiple axes is the product of the result per axis.

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
        The number of tiles each pixel would be covered by along an axis given the
        parameters.
    """
    return int(np.ceil((patch_size - overlap) / stride))


def compute_stride_for_mmse_count(
    patch_size: Sequence[int],
    overlap: Sequence[int],
    target_mmse_count: int,
    *,
    stride_z: int | None = None,
) -> tuple[tuple[int, int] | tuple[int, int, int], int]:
    """
    Compute the optimum stride in each spatial dimension to achieve a target MMSE count.

    The stride in X and Y will be equal.

    If the target MMSE count is not exactly achievable, the achieved MMSE count will
    always be greater than the target.

    Parameters
    ----------
    patch_size : Sequence[int]
        The input tile size, either 2D or 3D.
    overlap : Sequence[int]
        Describes the border cropped from each side of the tile, the border is
        `overlaps[dim] // 2` for each dimension `dim`.
    target_mmse_count : int
        Target MMSE count, e.g. how many tiles will cover each pixel.
    stride_z : int | None, default=None
        Optionally choose to manually fix the z_stride, ignored for 2D, if None for 3D
        it will be calculated. The effective MMSE count will be optimized by jointly
        tuning `stride_z` and the stride in xy.

    Returns
    -------
    tuple[int, int] | tuple[int, int, int]
        The calculated stride, 2D or 3D depending on the input.
    int
        The effective MMSE resulting from the calculated stride.

    Raises
    ------
    ValueError
        If the patch size is not square in XY.
    ValueError
        If the overlaps are not equal in XY.
    ValueError
        If the `patch_size` is not valid (not 2D or 3D).
    """
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
    """
    Find the stride that results in pixel coverage closest to the `target_mmse_count`.

    The stride is also subject to the following constraints:
    - The effective MMSE will always be greater than or equal to `target_mmse_count`,
    - The strides in X and Y are equal, and
    - If there are two solutions that result in the same optimum effective MMSE count
      then the result which minimizes the absolute difference between the stride in XY
      and the stride in Z is chosen.

    Parameters
    ----------
    crop_size_xy : int
        The size of the cropped tile in x and y after dropping the margins.
    crop_size_z : int
        The size of the cropped tile in z after dropping the margins.
    target_mmse_count : int
        The desired pixel coverage.

    Returns
    -------
    int
        The stride in XY.
    int
        The stride in Z.
    """

    def objective(value: tuple[int, int]) -> float:
        """Objective to minimize.

        Parameters
        ----------
        value : tuple[int, int]
            First value is the stride in the XY dimensions, second value is the stride
            in the Z dimension.

        Returns
        -------
        int
            Evaluated objective function.
        """
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

    `sum` and `count` share the same SC(Z)YX shape as the full image.
    """

    sum: NDArray[np.float32]
    count: NDArray[np.uint32]
    expected_tiles: int
    seen: int
    source: str
    axes: str
    original_data_shape: tuple[int, ...]

    def is_complete(self) -> bool:
        """Whether all tiles for this image have been accumulated.

        Returns
        -------
        bool
            Whether all tiles for this image have been accumulated.
        """
        return self.seen >= self.expected_tiles


def _allocate_accumulator(
    tile: ImageRegionData, output_channels: int
) -> _TileAccumulator:
    """Allocate an accumulator sized after the full image carried by `tile`.

    The spatial extent comes from `tile.data_shape`, while the channel dimension is
    overridden with `output_channels`.

    Parameters
    ----------
    tile : ImageRegionData
        A tile carrying input-image metadata, the `region_specs` attribute is a
        `TileSpecs` instance.
    output_channels : int
        Number of channels produced by the model.

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
        count=np.zeros(shape, dtype=np.uint32),
        expected_tiles=int(spec["total_tiles"]),
        seen=0,
        source=tile.source,
        axes=tile.axes,
        original_data_shape=tuple(int(d) for d in tile.original_data_shape),
    )


def _tile_paste_slices(
    spec: TileSpecs,
) -> tuple[
    tuple[slice | EllipsisType | int, ...], tuple[slice | EllipsisType | int, ...]
]:
    """Get slices to (i) crop the source tile and, (ii) paste to destination image.

    Parameters
    ----------
    spec : TileSpecs
        Tile specification that contains crop coordinates and stitching coordinates.

    Returns
    -------
    tuple[slice | EllipsisType | int, ...]
        The source slice that is used to crop the tile.
    tuple[slice | EllipsisType | int, ...]
        The slice that is used to paste the cropped tile into the destination image.
    """
    crop_coords = spec["crop_coords"]
    crop_size = spec["crop_size"]
    stitch_coords = spec["stitch_coords"]
    sample_idx = int(spec["sample_idx"])
    source: tuple[slice | EllipsisType | int, ...] = (
        ...,
        *[
            slice(int(c), int(c) + int(sz))
            for c, sz in zip(crop_coords, crop_size, strict=True)
        ],
    )
    dest: tuple[slice | EllipsisType | int, ...] = (
        sample_idx,
        ...,
        *[
            slice(int(s), int(s) + int(sz))
            for s, sz in zip(stitch_coords, crop_size, strict=True)
        ],
    )
    return source, dest


# TODO: should be a method on _TileAccumulator
def _paste_tile(
    tile_accumulator: _TileAccumulator, tile: ImageRegionData[TileSpecs]
) -> None:
    """Add a tile's cropped inner region into the accumulator and bump the count.

    Parameters
    ----------
    tile_accumulator : _TileAccumulator
        The tile accumulator for the image that the given `tile` belongs to.
    tile : ImageRegionData[TileSpecs]
        A predicted tile to paste into image.
    """
    spec: TileSpecs = tile.region_spec
    source_slice, dest_slice = _tile_paste_slices(spec)
    cropped = np.asarray(tile.data, dtype=np.float32)[source_slice]
    tile_accumulator.sum[dest_slice] += cropped
    tile_accumulator.count[dest_slice] += 1
    tile_accumulator.seen += 1


# TODO: should be a method on _TileAccumulator
def _finalize(tile_accumulator: _TileAccumulator, data_idx: int) -> NDArray[np.float32]:
    """Calculate the pixel-wise average of the overlapping tiles.

    The accumulated sum is divided by the count for each pixel.

    Parameters
    ----------
    tile_accumulator : _TileAccumulator
        The tile accumulator the corresponds to the image with `data_idx`.
    data_idx : int
        Denotes a specific image.

    Returns
    -------
    NDArray[np.float32]
        The resulting average.
    """
    if (tile_accumulator.count == 0).any():
        n_uncovered = int((tile_accumulator.count == 0).sum())
        logger.warning(
            "Image data_idx=%d has %d uncovered pixel(s). With "
            "SlidingWindowTiledPatching this should not happen — check your "
            "stride/overlap configuration. Those pixels will be returned as 0.",
            data_idx,
            n_uncovered,
        )
    mean = np.divide(
        tile_accumulator.sum,
        tile_accumulator.count,
        out=np.zeros_like(tile_accumulator.sum),
        where=tile_accumulator.count > 0,
        dtype=np.float32,
    )
    return mean


# TODO: why do we need this?
def _move_input_to_device(  # numpydoc ignore=PR01,RT01
    batch: tuple[ImageRegionData, ...], device: torch.device
) -> tuple[ImageRegionData, ...]:
    """Return `batch` with `batch[0].data` moved to `device`."""
    input_region = batch[0]
    moved = input_region._replace(data=input_region.data.to(device))  # type: ignore
    return (moved, *batch[1:])


def switi_prediction(
    model: MicroSplitModule,
    # TODO: change function to take data source and config to instantiate data loader.
    dataloader: DataLoader,
) -> tuple[list[NDArray], list[str]]:
    """Run SWITi inference.

    Parameters
    ----------
    model : MicroSplitModule
        Trained MicroSplit module. Must be initialized with
        `algorithm_config.mmse_count=1`.
    dataloader : DataLoader
        Prediction dataloader. Underlying dataset must use `SwitiPatching` strategy.

    Returns
    -------
    list of numpy.ndarray
        Per-image stitched predictions with axes `SC(Z)YX`.
    list of str
        Per-image sources (e.g. file paths), empty if input data is arrays.
    """
    # Model output channel count drives the stitch buffer's C axis.
    # For MicroSplit this differs from the input image's channel count.
    output_channels = int(model.config.model.output_channels)

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
            mean_region_batch = model.predict_step(batch, batch_idx)
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
