"""
Prediction convenience functions.

These functions are used during prediction.
"""
from typing import List

import numpy as np
import torch


def stitch_prediction(
    tiles: List[np.ndarray],
    stitching_data: List,
) -> np.ndarray:
    """
    Stitch tiles back together to form a full image.

    Parameters
    ----------
    tiles : List[Tuple[np.ndarray, List[int]]]
        Cropped tiles and their respective stitching coordinates.
    stitching_data : List
        List of coordinates obtained from
        dataset.tiling.compute_crop_and_stitch_coords_1d.

    Returns
    -------
    np.ndarray
        Full image.
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


def tta_forward(x: torch.Tensor) -> List[torch.Tensor]:
    """
    Augment 8-fold an array.

    The augmentation is performed using all 90 deg rotations and their flipped version,
    as well as the original image flipped.

    Tensors should be of shape SC(Z)YX, with S and C potentially singleton dimensions.

    Parameters
    ----------
    x : torch.Tensor
        Data to augment.

    Returns
    -------
    List
        Stack of augmented images.
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


def tta_backward(x_aug: List[torch.Tensor]) -> np.ndarray:
    """
    Invert `tta_forward` and average the 8 images.

    The function takes a list of torch tensors and returns a numpy array.

    Parameters
    ----------
    x_aug : List[torch.Tensor]
        Stack of 8-fold augmented images.

    Returns
    -------
    np.ndarray
        Average of de-augmented x_aug.
    """
    x_deaug = [
        x_aug[0].numpy(),
        np.rot90(x_aug[1], -1, axes=(2, 3)),
        np.rot90(x_aug[2], -2, axes=(2, 3)),
        np.rot90(x_aug[3], -3, axes=(2, 3)),
        np.flip(x_aug[4].numpy(), axis=(1, 3)),
        np.rot90(np.flip(x_aug[5].numpy(), axis=(1, 3)), -1, axes=(2, 3)),
        np.rot90(np.flip(x_aug[6].numpy(), axis=(1, 3)), -2, axes=(2, 3)),
        np.rot90(np.flip(x_aug[7].numpy(), axis=(1, 3)), -3, axes=(2, 3)),
    ]
    return np.mean(x_deaug, 0)
