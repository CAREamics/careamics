import itertools
from typing import Any, Generator, Optional

import numpy as np
from numpy.typing import NDArray

from careamics.config.tile_information import TileInformation


def extract_tiles(
    arr: NDArray,
    tile_size: tuple[int, ...],
    overlaps: tuple[int, ...],
    padding_kwargs: Optional[dict[str, Any]] = None,
) -> Generator[tuple[NDArray, TileInformation], None, None]:

    if padding_kwargs is None:
        padding_kwargs = {"mode": "refect"}

    data_shape = arr.shape

    # add padding
    padding = compute_padding(
        np.array(data_shape), np.array(tile_size), np.array(overlaps)
    )
    arr = np.pad(arr, padding, **padding_kwargs)

    # split array
    spatial_dims_shape = data_shape[-len(tile_size) :]
    grid_coords_iter = itertools.product(
        *[
            range(0, spatial_dims_shape[i], tile_size[i] - overlaps[i])
            for i in range(len(tile_size))
        ]
    )
    for tile_grid_coords in grid_coords_iter:
        crop_slices = [slice(i, i + tile_size) for i in tile_grid_coords]
        tile = arr[(..., *crop_slices)]

        tile_info = compute_tile_info(tile_grid_coords, data_shape, tile_size, overlaps)
        # TODO: kinda weird this is a generator,
        #   -> doesn't really save memory ? Don't think there are any places the tiles
        #    are not exracted all at the same time.
        #   Although I guess it would make sense for a zarr tile extractor.
        yield tile, tile_info


def compute_tile_info(
    tile_grid_coords: NDArray[np.int_],
    data_shape: NDArray[np.int_],
    tile_size: NDArray[np.int_],
    overlaps: NDArray[np.int_],
) -> TileInformation:

    spatial_dims_shape = data_shape[-len(tile_size) :]

    stitch_size = tile_size - overlaps
    stitch_coords_start = tile_grid_coords * stitch_size
    stitch_coords_end = stitch_coords_start + stitch_size

    tile_coords_start = stitch_coords_start - overlaps // 2

    # --- replace out of bounds indices
    out_of_lower_bound = stitch_coords_start < 0
    out_of_upper_bound = stitch_coords_end > spatial_dims_shape
    stitch_coords_start[out_of_lower_bound] = 0
    stitch_coords_end[out_of_upper_bound] = spatial_dims_shape[out_of_upper_bound]

    # --- calculate overlap crop coords
    overlap_crop_coords_start = stitch_coords_start - tile_coords_start
    overlap_crop_coords_end = overlap_crop_coords_start + (
        stitch_coords_end - stitch_coords_start
    )

    # --- combine start and end
    stitch_coords = tuple(
        (start, end) for start, end in zip(stitch_coords_start, stitch_coords_end)
    )
    overlap_crop_coords = tuple(
        (start, end)
        for start, end in zip(overlap_crop_coords_start, overlap_crop_coords_end)
    )

    tile_grid_shape = np.array(
        _compute_tile_grid_shape(data_shape, tile_size, overlaps)
    )
    last_tile = (tile_grid_coords == (tile_grid_shape - 1)).all()

    tile_info = TileInformation(
        array_shape=data_shape[1:],  # ignore S dim
        last_tile=last_tile,
        overlap_crop_coords=overlap_crop_coords,
        stitch_coords=stitch_coords,
        sample_id=0,  # TODO: in iterable dataset this is also always 0 pretty sure
    )
    return tile_info


def compute_padding(
    data_shape: NDArray[np.int_],
    tile_size: NDArray[np.int_],
    overlaps: NDArray[np.int_],
) -> tuple[int, int]:

    tile_grid_shape = np.array(
        _compute_tile_grid_shape(data_shape, tile_size, overlaps)
    )
    covered_shape = (tile_size - overlaps) * tile_grid_shape + overlaps

    pad_before = overlaps // 2
    pad_after = covered_shape - data_shape - pad_before

    return tuple((before, after) for before, after in zip(pad_before, pad_after))


def n_tiles_1d(axis_size: int, tile_size: int, overlap: int) -> int:
    return int(np.ceil(axis_size / (tile_size - overlap)))


def total_n_tiles(
    data_shape: tuple[int, ...], tile_size: tuple[int, ...], overlaps: tuple[int, ...]
) -> int:

    result = 1
    # assume spatial dimension are the last dimensions so iterate backwards
    for i in range(-1, -len(tile_size) - 1, -1):
        result = result * n_tiles_1d(data_shape[i], tile_size[i], overlaps[i])

    return result


def _compute_tile_grid_shape(
    data_shape: NDArray[np.int_],
    tile_size: NDArray[np.int_],
    overlaps: NDArray[np.int_],
) -> tuple[int, ...]:
    shape = [0 for _ in range(len(tile_size))]
    # assume spatial dimension are the last dimensions so iterate backwards
    for i in range(-1, -len(tile_size) - 1, -1):
        shape[i] = n_tiles_1d(data_shape[i], tile_size[i], overlaps[i])
    return tuple(shape)
