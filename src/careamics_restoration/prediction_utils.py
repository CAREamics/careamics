from typing import List

import numpy as np
import torch


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


def tta_forward(x: np.ndarray) -> List:
    """
    Augments x 8-fold: all 90 deg rotations plus lr flip of the four rotated versions.

    Parameters
    ----------
    x: torch.tensor
        data to augment

    Returns
    -------
    Stack of augmented x.
    """
    x_aug = [
        x,
        torch.rot90(x, 1, dims=(2, 3)),
        torch.rot90(x, 2, dims=(2, 3)),
        torch.rot90(x, 3, dims=(2, 3)),
    ]
    x_aug_flip = x_aug.copy()
    for x_ in x_aug:
        x_aug_flip.append(torch.flip(x_, dims=(1, 3)))
    return x_aug_flip


def tta_backward(x_aug: List) -> np.ndarray:
    """
    Inverts `tta_forward` and averages the 8 images.

    Parameters
    ----------
    x_aug: List
        stack of 8-fold augmented images.

    Returns
    -------
    average of de-augmented x_aug.
    """
    x_deaug = [
        x_aug[0],
        np.rot90(x_aug[1], -1),
        np.rot90(x_aug[2], -2),
        np.rot90(x_aug[3], -3),
        np.fliplr(x_aug[4]),
        np.rot90(np.fliplr(x_aug[5]), -1),
        np.rot90(np.fliplr(x_aug[6]), -2),
        np.rot90(np.fliplr(x_aug[7]), -3),
    ]
    return np.mean(x_deaug, 0)
