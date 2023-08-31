from typing import List

import numpy as np


def stitch_prediction(
    tiles: List[np.ndarray],
    stitching_data: List,
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
    # Get whole sample shape
    input_shape = stitching_data[0][0]
    predicted_image = np.zeros(input_shape, dtype=np.float32)
    for tile, (_, overlap_crop_coords, stitch_coords) in zip(tiles, stitching_data):
        # Compute coordinates for cropping predicted tile
        slices = tuple([slice(c[0], c[1]) for c in overlap_crop_coords])

        # Crop predited tile according to overlap coordinates
        cropped_tile = tile.squeeze()[slices]

        # Insert cropped tile into predicted image using stitch coordinates
        predicted_image[
            (..., *[slice(c[0], c[1]) for c in stitch_coords])
        ] = cropped_tile
    return predicted_image
