"""Collate function for tiling."""

from typing import Any, List, Tuple

import numpy as np
from torch.utils.data.dataloader import default_collate

from careamics.config.tile_information import TileInformation


def collate_tiles(batch: List[Tuple[np.ndarray, TileInformation]]) -> Any:
    """
    Collate tiles received from CAREamics prediction dataloader.

    CAREamics prediction dataloader returns tuples of arrays and TileInformation. In
    case of non-tiled data, this function will return the arrays. In case of tiled data,
    it will return the arrays, the last tile flag, the overlap crop coordinates and the
    stitch coordinates.

    Parameters
    ----------
    batch : List[Tuple[np.ndarray, TileInformation], ...]
        Batch of tiles.

    Returns
    -------
    Any
        Collated batch.
    """
    first_tile_info: TileInformation = batch[0][1]
    # if not tiled, then return arrays
    if not first_tile_info.tiled:
        arrays, _ = zip(*batch)

        return default_collate(arrays)
    # else we explicit the last_tile flag and coordinates
    else:
        new_batch = [tile for tile, _ in batch]
        tiles_batch = [tile_info for _, tile_info in batch]

        return default_collate(new_batch), tiles_batch
