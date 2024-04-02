import itertools
from typing import Generator, List, Tuple, Union

import numpy as np


def _compute_crop_and_stitch_coords_1d(
    axis_size: int, tile_size: int, overlap: int
) -> Tuple[List[Tuple[int, int]], ...]:
    """
    Compute the coordinates of each tile along an axis, given the overlap.

    Parameters
    ----------
    axis_size : int
        Length of the axis.
    tile_size : int
        Size of the tile for the given axis.
    overlap : int
        Size of the overlap for the given axis.

    Returns
    -------
    Tuple[Tuple[int]]
        Tuple of all coordinates for given axis.
    """
    # Compute the step between tiles
    step = tile_size - overlap
    crop_coords = []
    stitch_coords = []
    overlap_crop_coords = []

    # Iterate over the axis with step
    for i in range(0, max(1, axis_size - overlap), step):

        # Check if the tile fits within the axis
        if i + tile_size <= axis_size:
            
            # Add the coordinates to crop one tile
            crop_coords.append((i, i + tile_size))
            
            # Add the pixel coordinates of the cropped tile in the original image space
            stitch_coords.append(
                (
                    i + overlap // 2 if i > 0 else 0,
                    i + tile_size - overlap // 2
                    if crop_coords[-1][1] < axis_size
                    else axis_size,
                )
            )
            
            # Add the coordinates to crop the overlap from the prediction.
            overlap_crop_coords.append(
                (
                    overlap // 2 if i > 0 else 0,
                    tile_size - overlap // 2
                    if crop_coords[-1][1] < axis_size
                    else tile_size,
                )
            )

        # If the tile does not fit within the axis, perform the abovementioned
        # operations starting from the end of the axis
        else:
            # if (axis_size - tile_size, axis_size) not in crop_coords:
            crop_coords.append((max(0, axis_size - tile_size), axis_size))
            last_tile_end_coord = stitch_coords[-1][1] if stitch_coords else 1
            stitch_coords.append((last_tile_end_coord, axis_size))
            overlap_crop_coords.append(
                (tile_size - (axis_size - last_tile_end_coord), tile_size)
            )
            break
    return crop_coords, stitch_coords, overlap_crop_coords


def extract_tiles(
    arr: np.ndarray,
    tile_size: Union[List[int], Tuple[int, ...]],
    overlaps: Union[List[int], Tuple[int, ...]],
) -> Generator[
    Tuple[np.ndarray, bool, Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]], 
    None, 
    None
]:
    """
    Generate tiles from the input array with specified overlap.

    The tiles cover the whole array. The method returns a generator that yields a tuple
    containing the following:

    - tile: np.ndarray, shape (C, (Z), Y, X).
    - last_tile: bool, whether this is the last tile.
    - shape: Tuple[int], shape of a tile, excluding the S dimension.
    - overlap_crop_coords: Tuple[int], coordinates used to crop the tile during
        stitching.
    - stitch_coords: Tuple[int], coordinates used to stitch the tiles back to the full
        image.

    Parameters
    ----------
    arr : np.ndarray
        Array of shape (S, C, (Z), Y, X).
    tile_size : Union[List[int], Tuple[int]]
        Tile sizes in each dimension, of length 2 or 3.
    overlaps : Union[List[int], Tuple[int]]
        Overlap values in each dimension, of length 2 or 3.

    Yields
    ------
    Generator[Tuple[np.ndarray, bool, Tuple[int], np.ndarray, np.ndarray], None, None]        
        Generator of tuple containing the tile, last tile boolean, array shape, 
        overlap_crop_coords, and stitching coords.
    """
    # Iterate over num samples (S)
    for sample_idx in range(arr.shape[0]):
        sample: np.ndarray = arr[sample_idx, ...]

        # Create a list of coordinates for cropping and stitching all axes.
        # [crop coordinates, stitching coordinates, overlap crop coordinates]
        # For axis of size 35 and patch size of 32 compute_crop_and_stitch_coords_1d
        # will output ([(0, 32), (3, 35)], [(0, 20), (20, 35)], [(0, 20), (17, 32)])
        crop_and_stitch_coords_list = [
            _compute_crop_and_stitch_coords_1d(
                sample.shape[i + 1], tile_size[i], overlaps[i]
            )
            for i in range(len(tile_size))
        ]

        # Rearrange crop coordinates from a list of coordinate pairs per axis to a list
        # grouped by type.
        all_crop_coords, all_stitch_coords, all_overlap_crop_coords = zip(
            *crop_and_stitch_coords_list
        )

        # Maximum tile index
        max_tile_idx = np.prod([len(axis) for axis in all_crop_coords]) - 1

        # Iterate over generated coordinate pairs:
        for tile_idx, (crop_coords, stitch_coords, overlap_crop_coords) in enumerate(
            zip(
                itertools.product(*all_crop_coords),
                itertools.product(*all_stitch_coords),
                itertools.product(*all_overlap_crop_coords),
            )
        ):
            # Extract tile from the sample
            tile: np.ndarray = sample[
                (..., *[slice(c[0], c[1]) for c in list(crop_coords)]) # type: ignore
            ]

            # Check if we are at the end of the sample by computing the length of the 
            # array that contains all the tiles
            if tile_idx == max_tile_idx:
                last_tile = True
            else:
                last_tile = False
            
            yield (
                tile.astype(np.float32),
                last_tile,
                arr.shape[1:], # TODO is this used anywhere??
                overlap_crop_coords,
                stitch_coords,
            )
