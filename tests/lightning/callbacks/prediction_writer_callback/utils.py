"""Utility functions to be used in tests relating to the `PredictionWriterCallback`."""

import numpy as np
from numpy.typing import NDArray

from careamics.config.tile_information import TileInformation
from careamics.dataset.tiling import extract_tiles
from careamics.lightning.callbacks.prediction_writer_callback.write_strategy import (
    caches,
)


def patch_tile_cache(
    tile_cache: caches.TileCache,
    tiles: list[NDArray],
    tile_infos: list[TileInformation],
) -> None:
    """
    Patch simulated tile cache into `strategy`.

    Parameters
    ----------
    tile_cache : TileCache
        Tile cache used in `WriteTiles` write strategy class.
    tiles : list of NDArray
        Tiles to patch into `strategy.tile_cache`.
    tile_infos : list of TileInformation
        Corresponding tile information to patch into `strategy.tile_info_cache`.
    """
    tile_cache.add((np.concatenate(tiles), tile_infos))


def create_tiles(n_samples: int) -> tuple[list[NDArray], list[TileInformation]]:
    """
    Create a set of tiles from `n_samples`.

    To create the tiles the following parameters, `tile_size=(4, 4)` and
    `overlaps=(2, 2)`, on an input array with shape (`n_samples`, 1, 8, 8); this
    results in 9 tiles per sample.

    Parameters
    ----------
    n_samples : int
        Number of samples to simulate the tiles from.

    Returns
    -------
    tuple of (list of NDArray), list of TileInformation))
        Tuple where first element is the list of tiles and second element is a list
        of corresponding tile information.
    """

    input_shape = (n_samples, 1, 8, 8)
    tile_size = (4, 4)
    tile_overlap = (2, 2)

    arr = np.arange(np.prod(input_shape)).reshape(input_shape)

    all_tiles = list(extract_tiles(arr, tile_size, tile_overlap))
    tiles = [output[0] for output in all_tiles]
    tile_infos = [output[1] for output in all_tiles]

    return tiles, tile_infos
