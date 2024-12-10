from typing import Callable
from unittest.mock import Mock

import numpy as np
import pytest
from numpy.typing import NDArray

from careamics.config.tile_information import TileInformation
from careamics.dataset.tiling import extract_tiles
from careamics.file_io import WriteFunc
from careamics.lightning.callbacks.prediction_writer_callback.write_strategy import (
    caches,
)


@pytest.fixture
def write_func():
    """Mock `WriteFunc`."""
    return Mock(spec=WriteFunc)


@pytest.fixture
def patch_tile_cache() -> (
    Callable[[caches.TileCache, list[NDArray], list[TileInformation]], None]
):
    def inner(
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

    return inner


@pytest.fixture
def create_tiles() -> Callable[[int], tuple[list[NDArray], list[TileInformation]]]:
    def inner(n_samples: int) -> tuple[list[NDArray], list[TileInformation]]:
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

    return inner


@pytest.fixture
def samples() -> tuple[NDArray, list[int]]:
    n_samples_per_file = [3, 1, 2]
    shapes = [(16, 16), (8, 8), (12, 12)]
    sample_set = []
    for n_samples, spatial_shape in zip(n_samples_per_file, shapes):
        shape = (n_samples, 1, *spatial_shape)
        sample = np.arange(np.prod(shape)).reshape(shape)
        sample_set.extend(np.split(sample, n_samples))

    return sample_set, n_samples_per_file
