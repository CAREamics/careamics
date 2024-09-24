
import numpy as np
from numpy.typing import NDArray

from careamics.config.tile_information import TileInformation

class TileCache:
    """
    Cache tiles; logic to pop tiles when tiles from a full image have been stored.
    """

    def __init__(self):
        self.array_cache: list[NDArray] = []
        self.tile_info_cache: list[TileInformation] = []

    def add(self, item: tuple[NDArray, list[TileInformation]]):
        self.array_cache.extend(np.split(item[0]), item[0].shape[0])
        self.tile_info_cache.extend(item[1])

    def has_last_tile(self) -> bool:
        return any(tile_info.last_tile for tile_info in self.tile_info_cache)
    
    def pop_image_tiles(self) -> tuple[list[NDArray], list[TileInformation]]:
        is_last_tile = [tile_info.last_tile for tile_info in self.tile_info_cache]
        if not any(is_last_tile):
            raise ValueError("No last tile in cache.")
        
        index = np.where(is_last_tile)[0][0]
        # get image tiles
        tiles = self.array_cache[: index + 1]
        tile_infos = self.tile_info_cache[: index + 1]
        # remove image tiles from list
        self.array_cache = self.array_cache[index + 1 :]
        self.tile_info_cache = self.tile_info_cache[index + 1 :]

        return tiles, tile_infos
    
    def reset(self):
        self.array_cache = []
        self.tile_info_cache = []