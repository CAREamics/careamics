"""Tiled prediction stitching utilities."""

import builtins
from collections import defaultdict
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from careamics.dataset_ng.dataset import ImageRegionData
from careamics.dataset_ng.patching_strategies import TileSpecs


def group_tiles_by_key(
    tiles: list[ImageRegionData], key: Literal["data_idx", "sample_idx"]
) -> dict[int, list[ImageRegionData]]:
    """
    Sort tiles by key.

    Parameters
    ----------
    tiles : list of ImageRegionData
        List of tiles to sort.
    key : {'data_idx', 'sample_idx'}
        Key to group tiles by.

    Returns
    -------
    {int: list of ImageRegionData}
        Dictionary mapping data indices to lists of tiles.
    """
    sorted_tiles: dict[int, list[ImageRegionData]] = defaultdict(list)
    for tile in tiles:
        key_value = tile.region_spec[key]
        sorted_tiles[key_value].append(tile)
    return sorted_tiles


def stitch_prediction(
    tiles: list[ImageRegionData],
) -> tuple[list[NDArray], list[str]]:
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
    list of str
        List of sources, one per output.
    """
    # sort tiles by data index
    grouped_tiles: dict[int, list[ImageRegionData]] = group_tiles_by_key(
        tiles, key="data_idx"
    )

    # stitch each image separately
    image_predictions: list[NDArray] = []
    image_sources: list[str] = []
    for data_idx in sorted(grouped_tiles.keys()):
        image_predictions.append(stitch_single_prediction(grouped_tiles[data_idx]))
        image_sources.append(grouped_tiles[data_idx][0].source)

    return image_predictions, image_sources


def stitch_single_prediction(
    tiles: list[ImageRegionData],
) -> NDArray:
    """
    Stitch tiles back together to form a full image.

    Tiles are of dimensions C(Z)YX, where C is the number of channels and can be a
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
        tiles_by_sample = group_tiles_by_key(tiles, key="sample_idx")
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
        # predicted_image has singleton sample dimension
        predicted_image[0] = stitch_single_sample(tiles)

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
    data_shape = tiles[0].data_shape  # SC(Z)YX
    predicted_sample = np.zeros(data_shape[1:], dtype=np.float32)

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
