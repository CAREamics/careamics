"""Prediction utility functions."""

from typing import List

import numpy as np

from careamics.config.tile_information import TileInformation


def stitch_prediction(
    tiles: List[np.ndarray],
    tile_infos: List[TileInformation],
) -> np.ndarray:
    """Stitch tiles back together to form a full image.

    Tiles are of dimensions SC(Z)YX, where C is the number of channels and can be a
    singleton dimension.

    Parameters
    ----------
    tiles : List[np.ndarray]
        Cropped tiles and their respective stitching coordinates.
    tile_infos : List[TileInformation]
        List of information and coordinates obtained from
        `dataset.tiled_patching.extract_tiles`.

    Returns
    -------
    np.ndarray
        Full image.
    """
    # retrieve whole array size
    input_shape = tile_infos[0].array_shape
    predicted_image = np.zeros(input_shape, dtype=np.float32)

    for tile, tile_info in zip(tiles, tile_infos):
        n_channels = tile.shape[0]

        # Compute coordinates for cropping predicted tile
        slices = (slice(0, n_channels),) + tuple(
            [slice(c[0], c[1]) for c in tile_info.overlap_crop_coords]
        )

        # Crop predited tile according to overlap coordinates
        cropped_tile = tile[slices]

        # Insert cropped tile into predicted image using stitch coordinates
        predicted_image[
            (
                ...,
                *[slice(c[0], c[1]) for c in tile_info.stitch_coords],
            )
        ] = cropped_tile.astype(np.float32)

    return predicted_image
