"""Sliding-window inner-tiled (SWITi) prediction.

Functions for stitching densely sampled overlapping tiles produced by the
`SWITiPatching` strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import EllipsisType
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from careamics.config.data import MicroSplitDataConfig
from careamics.config.data.patching_strategies import SwitiPatchingConfig
from careamics.dataset.factory import ImageStackLoading, ReadFuncLoading
from careamics.dataset.factory.microsplit_factory import create_microsplit_pred_dataset
from careamics.dataset.image_region_data import ImageRegionData
from careamics.dataset.patching import TileSpecs
from careamics.lightning.data import InputVar
from careamics.lightning.data.data_module_utils import initialize_data_pair
from careamics.lightning.modules.microsplit_module import MicroSplitModule
from careamics.lightning.prediction.convert_prediction import (
    decollate_image_region_data,
)
from careamics.utils import get_logger

logger = get_logger(__name__)


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


# TODO: add tests
def switi_prediction(
    model: MicroSplitModule,
    data_config: MicroSplitDataConfig,
    pred_data: InputVar,
    loading: ReadFuncLoading | ImageStackLoading | None = None,
) -> tuple[list[NDArray], list[str]]:
    """Run SWITi inference.

    Parameters
    ----------
    model : MicroSplitModule
        Trained MicroSplit module. Must be initialized with
        `algorithm_config.mmse_count=1`.
    data_config : MicroSplitDataConfig
        Data configuration for MicroSplit. Must have the "switi" patching strategy
        configuration.
    pred_data : pathlib.Path, str, numpy.ndarray, or sequence of these
        Data to predict on. Can be a single item or a sequence of paths/arrays.
    loading : Loading, default=None
        Loading strategy to use for the prediction data. May be a ReadFuncLoading or
        ImageStackLoading. If None, uses the loading strategy from the training
        configuration.

    Returns
    -------
    list of numpy.ndarray
        Per-image stitched predictions with axes `SC(Z)YX`.
    list of str
        Per-image sources (e.g. file paths), empty if input data is arrays.
    """
    # create dataset and dataloader
    if not isinstance(data_config.patching, SwitiPatchingConfig):
        raise TypeError(
            "Patching strategy in the provided `data_config` must be "
            f"`SwitiPatchingConfig`, got {type(data_config.patching)} instead."
        )
    # pred_data: Sequence[NDArray[Any]] | Sequence[Path]
    pred_data_validated, _ = initialize_data_pair(
        data_type=data_config.data_type, input_data=pred_data, loading=loading
    )
    dataset = create_microsplit_pred_dataset(
        config=data_config, input_data=pred_data_validated, loading=loading
    )
    dataloader = DataLoader(
        dataset,
        batch_size=dataset.config.batch_size,
        collate_fn=default_collate,
        **data_config.pred_dataloader_params,
    )

    # Model output channel count drives the stitch buffer's C axis.
    # For MicroSplit this differs from the input image's channel count.
    output_channels = int(model.config.model.output_channels)

    accumulators: dict[int, _TileAccumulator] = {}
    finalized: dict[int, tuple[NDArray[Any], str]] = {}

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

    # TODO: restore shape
    return predictions_output, sources
