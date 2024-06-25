"""Prediction utility functions."""

from typing import List

import numpy as np

from careamics.config.tile_information import TileInformation


# TODO: why not allow input and output of torch.tensor ?
def stitch_prediction(
    tiles: List[np.ndarray],
    tile_infos: List[TileInformation],
) -> List[np.ndarray]:
    """
    Stitch tiles back together to form a full image(s).

    Tiles are of dimensions SC(Z)YX, where C is the number of channels and can be a
    singleton dimension.

    Parameters
    ----------
    tiles : list of numpy.ndarray
        Cropped tiles and their respective stitching coordinates. Can contain tiles
        from multiple images.
    tile_infos : list of TileInformation
        List of information and coordinates obtained from
        `dataset.tiled_patching.extract_tiles`.

    Returns
    -------
    list of numpy.ndarray
        Full image(s).
    """
    # Find where to split the lists so that only info from one image is contained.
    # Do this by locating the last tiles of each image.
    last_tiles = [tile_info.last_tile for tile_info in tile_infos]
    last_tile_position = np.where(last_tiles)[0]
    image_slices = [
        slice(
            None if i == 0 else last_tile_position[i - 1] + 1, last_tile_position[i] + 1
        )
        for i in range(len(last_tile_position))
    ]
    image_predictions = []
    # slice the lists and apply stitch_prediction_single to each in turn.
    for image_slice in image_slices:
        image_predictions.append(
            stitch_prediction_single(tiles[image_slice], tile_infos[image_slice])
        )
    return image_predictions


def stitch_prediction_single(
    tiles: List[np.ndarray],
    tile_infos: List[TileInformation],
) -> np.ndarray:
    """
    Stitch tiles back together to form a full image.

    Tiles are of dimensions SC(Z)YX, where C is the number of channels and can be a
    singleton dimension.

    Parameters
    ----------
    tiles : list of numpy.ndarray
        Cropped tiles and their respective stitching coordinates.
    tile_infos : list of TileInformation
        List of information and coordinates obtained from
        `dataset.tiled_patching.extract_tiles`.

    Returns
    -------
    numpy.ndarray
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
