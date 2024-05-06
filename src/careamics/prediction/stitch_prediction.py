"""
Prediction convenience functions.

These functions are used during prediction.
"""

from typing import List

import numpy as np
import torch


def stitch_prediction(
    tiles: List[torch.Tensor],
    stitching_data: List[List[torch.Tensor]],
) -> torch.Tensor:
    """
    Stitch tiles back together to form a full image.

    Parameters
    ----------
    tiles : List[torch.Tensor]
        Cropped tiles and their respective stitching coordinates.
    stitching_coords : List
        List of information and coordinates obtained from
        `dataset.tiled_patching.extract_tiles`.

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
    predicted_image = torch.Tensor(np.zeros(input_shape, dtype=np.float32))

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
            ] = cropped_tile.to(torch.float32)

    return predicted_image
