import os
import torch
import logging
import itertools
import tifffile
import numpy as np

from functools import partial
from torch.utils.data import Dataset, IterableDataset, DataLoader
from pathlib import Path
from skimage.util import view_as_windows
from typing import Callable, List, Optional, Sequence, Union, Tuple


def extract_patches_predict(
    arr: np.ndarray, patch_size: Tuple[int], overlap: Tuple[int], is_time_series: bool
) -> List[np.ndarray]:
    # TODO remove hard coded vals
    # Overlap is half of the value mentioned in original N2V #TODO must be even. It's like this because of current N2V notation
    # TODO range start from 1, because 0 is the channel dimension
    # TODO check if patch size == image size
    if is_time_series:
        arr = arr.reshape(-1,  *arr.shape[2:])
    actual_overlap = [
        arr.shape[0],
        *[patch_size[i] - overlap[i - 1] for i in range(1, len(patch_size))],
    ]

    if len(patch_size) + 1 != len(actual_overlap):
        # TODO ugly fix for incosistent overlap shape
        actual_overlap.insert(0, 1)
    # TODO this is getting really ugly
    arr = arr.numpy()

    all_tiles = view_as_windows(
        arr, window_shape=[arr.shape[0], *patch_size], step=actual_overlap
    )  # shape (tiles in y, tiles in x, Y, X)
    # TODO properly handle 2d/3d, copy from sequential patch extraction
    # TODO questo e una grande cazzata !!!
    output_shape = (
        arr.shape[0],
        arr.shape[1],
        all_tiles.shape[2],
        all_tiles.shape[3],
        *patch_size[1:],
    )
    all_tiles = all_tiles.reshape(*output_shape)
    # TODO yet another ugly hardcode ! :len(patch_size)+1
    for tile_coords in itertools.product(
        *map(range, all_tiles.shape[: len(patch_size) + 1])
    ):  # TODO add 2/3d automatic selection of axes
        # TODO test for number of tiles in each category
        tile = all_tiles[(*[c for c in tile_coords], ...)]
        return (
            tile.astype(np.float32),
            tile_coords,
            all_tiles.shape[: len(patch_size) + 1],
            overlap,
        )


def calculate_stitching_coords(
    tile_coords: Tuple[int], last_tile_coord: Tuple[int], overlap: Tuple[int]
) -> Tuple[slice]:

    # TODO add 2/3d support
    # TODO different overlaps for each dimension
    # TODO different patch sizes for each dimension
    list_coord = []

    for i, coord in enumerate(tile_coords):
        if coord == 0:
            list_coord.append(slice(0, -overlap[i] // 2))
        elif coord == last_tile_coord[i] - 1:
            list_coord.append(slice(overlap[i] // 2, None))
        else:
            list_coord.append(slice(overlap[i] // 2, -overlap[i] // 2))
    return list_coord
