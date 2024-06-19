"""Prediction utility functions."""

from typing import List

import numpy as np


def stitch_prediction(
    tiles: List[np.ndarray],
    stitching_data: List[List[np.ndarray]],
) -> np.ndarray:
    """
    Stitch tiles back together to form a full image.

    Parameters
    ----------
    tiles : List[np.ndarray]
        Cropped tiles and their respective stitching coordinates.
    stitching_data : List[List[np.ndarray]]
        List of lists containing the overlap crop coordinates and stitch coordinates.

    Returns
    -------
    np.ndarray
        Full image.
    """
    # retrieve whole array size, there is two cases to consider:
    # 1. the tiles are stored in a list
    # 2. the tiles are stored in a list with batches along the first dim
    if tiles[0].shape[0] > 1:
        input_shape = np.array(
            [el.numpy() for el in stitching_data[0][0][0]], dtype=int
        ).squeeze()
    else:
        input_shape = np.array(
            [el.numpy() for el in stitching_data[0][0]], dtype=int
        ).squeeze()

    # TODO should use torch.zeros instead of np.zeros
    predicted_image = np.zeros(input_shape, dtype=np.float32)

    for tile_batch, (_, overlap_crop_coords_batch, stitch_coords_batch) in zip(
        tiles, stitching_data
    ):
        for batch_idx in range(tile_batch.shape[0]):
            # Compute coordinates for cropping predicted tile
            slices = tuple(
                [
                    slice(c[0][batch_idx], c[1][batch_idx])
                    for c in overlap_crop_coords_batch
                ]
            )

            # Crop predited tile according to overlap coordinates
            cropped_tile = tile_batch[batch_idx].squeeze()[slices]

            # Insert cropped tile into predicted image using stitch coordinates
            predicted_image[
                (
                    ...,
                    *[
                        slice(c[0][batch_idx], c[1][batch_idx])
                        for c in stitch_coords_batch
                    ],
                )
            ] = cropped_tile.astype(np.float32)

    return predicted_image
