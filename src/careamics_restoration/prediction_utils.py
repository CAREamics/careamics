from typing import List, Tuple

import numpy as np
import torch


def stitch_prediction(
    tiles: List[Tuple[np.ndarray, List[torch.tensor]]], input_shape: Tuple[int]
) -> np.ndarray:
    """Stitches tiles back together to form a full image.

    Parameters
    ----------
    tiles : List[Tuple[np.ndarray, List[int]]]
        Tuple of cropped tiles and their respective stitching coordinates
    input_shape : Tuple[int]
        Shape of the full image

    Returns
    -------
    np.ndarray
        Full image
    """
    predicted_image = np.zeros(input_shape, dtype=np.float32)
    for tile, tile_coords in tiles:
        predicted_image[(..., *[slice(c[0], c[1]) for c in tile_coords])] = tile
    return predicted_image
