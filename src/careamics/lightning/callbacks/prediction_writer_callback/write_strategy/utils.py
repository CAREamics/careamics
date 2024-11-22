from typing import Optional

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
        self.array_cache.extend(np.split(item[0], item[0].shape[0]))
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


class SampleCache:

    def __init__(self, n_samples_per_file: list[int]):

        self.n_samples_per_file: list[int] = n_samples_per_file
        self.n_samples_iter = iter(self.n_samples_per_file)
        # n_samples will be set to None once iterated through each element
        self.n_samples: Optional[int] = next(self.n_samples_iter)
        self.sample_cache: list[NDArray] = []

    def add(self, item: NDArray):
        self.sample_cache.extend(np.split(item, item.shape[0]))

    def has_all_file_samples(self) -> bool:
        if self.n_samples is None:
            raise ValueError(
                "Number of samples for current file is unknown. Reached the end of the "
                "given list of samples per file, or a list has not been given."
            )
        return len(self.sample_cache) >= self.n_samples

    def pop_file_samples(self) -> list[NDArray]:
        if not self.has_all_file_samples():
            raise ValueError(
                "Do not have all the samples belonging to the current file."
            )

        samples = self.sample_cache[: self.n_samples]
        self.sample_cache = self.sample_cache[self.n_samples :]

        try:
            self.n_samples = next(self.n_samples_iter)
        except StopIteration:
            self.n_samples = None

        return samples

    def reset(self):
        self.n_samples_iter = iter(self.n_samples_per_file)
        self.sample_cache: list[NDArray] = []
