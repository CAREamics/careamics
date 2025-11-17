"""Tiled prediction stitching utilities."""

import builtins
from collections import defaultdict

import numpy as np
from numpy.typing import NDArray

from careamics.dataset_ng.dataset import ImageRegionData
from careamics.dataset_ng.patching_strategies import TileSpecs


def sort_tiles_by_data_index(
    tiles: list[ImageRegionData],
) -> dict[int, list[ImageRegionData]]:
    """
    Sort tiles by their data index.

    Parameters
    ----------
    tiles : list of ImageRegionData
        List of tiles to sort.

    Returns
    -------
    {int: list of ImageRegionData}
        Dictionary mapping data indices to lists of tiles.
    """
    sorted_tiles: dict[int, list[ImageRegionData]] = defaultdict(list)
    for tile in tiles:
        data_idx = tile.region_spec["data_idx"]
        sorted_tiles[data_idx].append(tile)
    return sorted_tiles


def sort_tiles_by_sample_index(
    tiles: list[ImageRegionData],
) -> dict[int, list[ImageRegionData]]:
    """
    Sort tiles by their sample index.

    Parameters
    ----------
    tiles : list of ImageRegionData
        List of tiles to sort.

    Returns
    -------
    {int: list of ImageRegionData}
        Dictionary mapping sample indices to lists of tiles.
    """
    sorted_tiles: dict[int, list[ImageRegionData]] = defaultdict(list)
    for tile in tiles:
        sample_idx = tile.region_spec["sample_idx"]
        sorted_tiles[sample_idx].append(tile)
    return sorted_tiles


def stitch_prediction(
    tiles: list[ImageRegionData],
) -> list[NDArray]:
    """
    Stitch tiles back together to form full images.

    Tiles are of dimensions SC(Z)YX, where C is the number of channels and can be a
    singleton dimension.

    Parameters
    ----------
    tiles : list of ImageRegionData
        Cropped tiles and their respective stitching coordinates. Can contain tiles
        from multiple images.

    Returns
    -------
    list of numpy.ndarray
        Full images, may be a single image.
    """
    # sort tiles by data index
    sorted_tiles: dict[int, list[ImageRegionData]] = sort_tiles_by_data_index(tiles)

    # stitch each image separately
    image_predictions: list[NDArray] = []
    for data_idx in sorted_tiles.keys():
        image_predictions.append(stitch_single_prediction(sorted_tiles[data_idx]))
    return image_predictions


def stitch_single_prediction(
    tiles: list[ImageRegionData],
) -> NDArray:
    """
    Stitch tiles back together to form a full image.

    Tiles are of dimensions SC(Z)YX, where C is the number of channels and can be a
    singleton dimension.

    Parameters
    ----------
    tiles : list of ImageRegionData
        Cropped tiles and their respective stitching coordinates.

    Returns
    -------
    numpy.ndarray
        Full image, with dimensions SC(Z)YX.
    """
    data_shape = tiles[0].data_shape
    predicted_image = np.zeros(data_shape, dtype=np.float32)

    if "S" in tiles[0].axes:
        tiles_by_sample = sort_tiles_by_sample_index(tiles)
        for sample_idx in tiles_by_sample.keys():
            sample_tiles = tiles_by_sample[sample_idx]
            stitched_sample = stitch_single_sample(sample_tiles)

            # compute sample slice
            sample_slice = slice(
                sample_idx,
                sample_idx + 1,
            )

            # insert stitched sample into predicted image
            predicted_image[sample_slice] = stitched_sample.astype(np.float32)
    else:
        # stitch as a single sample
        predicted_image = stitch_single_sample(tiles)

    return predicted_image


def stitch_single_sample(
    tiles: list[ImageRegionData],
) -> NDArray:
    """
    Stitch tiles back together to form a full sample.

    Tiles are of dimensions C(Z)YX, where C is the number of channels and can be a
    singleton dimension.

    Parameters
    ----------
    tiles : list of ImageRegionData
        Cropped tiles and their respective stitching coordinates.

    Returns
    -------
    numpy.ndarray
        Full sample, with dimensions C(Z)YX.
    """
    data_shape = tiles[0].data_shape
    len_patches = len(tiles[0].data.squeeze().shape)

    predicted_sample = np.zeros(data_shape[-len_patches:], dtype=np.float32)

    for tile in tiles:
        # compute crop coordinates and stitiching coordinates
        tile_spec: TileSpecs = tile.region_spec  # type: ignore
        crop_coords = tile_spec["crop_coords"]
        crop_size = tile_spec["crop_size"]
        stitch_coords = tile_spec["stitch_coords"]

        crop_slices: tuple[builtins.ellipsis | slice, ...] = (
            ...,
            *[
                slice(start, start + length)
                for start, length in zip(crop_coords, crop_size, strict=True)
            ],
        )

        stitch_slices: tuple[builtins.ellipsis | slice, ...] = (
            ...,
            *[
                slice(start, start + length)
                for start, length in zip(stitch_coords, crop_size, strict=True)
            ],
        )

        # crop predited tile according to overlap coordinates
        cropped_tile = tile.data[crop_slices]

        # insert cropped tile into predicted image
        predicted_sample[stitch_slices] = cropped_tile.astype(np.float32)

    return predicted_sample
