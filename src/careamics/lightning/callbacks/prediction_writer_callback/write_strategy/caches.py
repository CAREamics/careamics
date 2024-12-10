"""Utility classes, for caching data, used in the write strategies."""

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from careamics.config.tile_information import TileInformation


class TileCache:
    """
    Logic to cache tiles, then pop tiles when tiles from a full image have been stored.

    Attributes
    ----------
    array_cache : list[numpy.ndarray]
        The tile arrays with the dimensions SC(Z)YX.
    tile_info_cache : list[TileInformation]
        The corresponding tile information for each tile.
    """

    def __init__(self):
        """Logic to cache tiles, and pop tiles when a full set of have been stored."""
        self.array_cache: list[NDArray] = []
        self.tile_info_cache: list[TileInformation] = []

    def add(self, item: tuple[NDArray, list[TileInformation]]):
        """
        Add another batch to the cache.

        Parameters
        ----------
        item : tuple of (numpy.ndarray, list[TileInformation])
            Tuple where the first element is a concatenated set of tiles, and the
            second element is a list of each corresponding `TileInformation`.
        """
        self.array_cache.extend(np.split(item[0], item[0].shape[0]))
        self.tile_info_cache.extend(item[1])

    def has_last_tile(self) -> bool:
        """
        Determine whether the current cache contains the last tile of a sample.

        Returns
        -------
        bool
            Whether the last tile is contained in the cache.
        """
        return any(tile_info.last_tile for tile_info in self.tile_info_cache)

    def pop_image_tiles(self) -> tuple[list[NDArray], list[TileInformation]]:
        """
        Pop the tiles that will create a full image from the cache.

        I.e. The tiles belonging to a full image will be removed from the cache but
        returned by this function call.

        Returns
        -------
        list of numpy.ndarray
            A list of tiles with the dimensions SC(Z)YX.
        list of TileInformation
            A list of corresponding tile information.

        Raises
        ------
        ValueError
            If the tiles belonging to a full image are not contained in the cache, i.e.
            if the cache does not contain the last tile of an image.
        """
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
        """Reset the cache. Remove all tiles and tile information from the cache."""
        self.array_cache = []
        self.tile_info_cache = []


class SampleCache:
    """
    Logic to cache samples until they can be concatenated together to create a file.

    Parameters
    ----------
    n_samples_per_file : list[int]
        A list that contains the number of samples that will be contained in each
        file. There should be `n` elements in the list for `n` files intended to
        be created.
    """

    def __init__(self, n_samples_per_file: list[int]):
        """
        Logic to cache samples until they can be concatenated together to create a file.

        Parameters
        ----------
        n_samples_per_file : list[int]
            A list that contains the number of samples that will be contained in each
            file. There should be `n` elements in the list for `n` files intended to
            be created.
        """
        self.n_samples_per_file: list[int] = n_samples_per_file
        self.n_samples_iter = iter(self.n_samples_per_file)
        # n_samples will be set to None once iterated through each element
        self.n_samples: Optional[int] = next(self.n_samples_iter)
        self.sample_cache: list[NDArray] = []

    def add(self, item: NDArray):
        """
        Add a sample to the cache.

        Parameters
        ----------
        item : numpy.ndarray
            A set of predicted samples.
        """
        self.sample_cache.extend(np.split(item, item.shape[0]))

    def has_all_file_samples(self) -> bool:
        """
        Determine if all the samples for the current file are contained in the cache.

        Returns
        -------
        bool
            Whether all the samples are contained in the cache.
        """
        if self.n_samples is None:
            raise ValueError(
                "Number of samples for current file is unknown. Reached the end of the "
                "given list of samples per file, or a list has not been given."
            )
        return len(self.sample_cache) >= self.n_samples

    def pop_file_samples(self) -> list[NDArray]:
        """
        Pop from the cache the samples required for the current file to be created.

        Returns
        -------
        list of numpy.ndarray
            A list of samples to concatenate together into a file.
        """
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
        """Reset the cache. Remove all the samples from the cache."""
        self.n_samples_iter = iter(self.n_samples_per_file)
        self.sample_cache: list[NDArray] = []
