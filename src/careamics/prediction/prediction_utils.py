"""
Prediction convenience functions.

These functions are used during prediction.
"""
from typing import List, Optional

import numpy as np
import torch


def stitch_prediction(
    tiles: List[torch.Tensor],
    stitching_data: List,
    explicit_stitching: Optional[bool] = False,
) -> torch.Tensor:
    """
    Stitch tiles back together to form a full image.

    Parameters
    ----------
    tiles : List[torch.Tensor]
        Cropped tiles and their respective stitching coordinates.
    stitching_data : List
        List of coordinates obtained from
        `dataset.tiled_patching.extract_tiles`.
    explicit_stitching : bool, optional
        Whether this function is called explicitly after prediction(Lighting) or inside
        the predict function. Removes the first element (last tile indicator).

    Returns
    -------
    np.ndarray
        Full image.
    """
    # Remove first element of stitching_data if explicit_stitching
    # TODO revisit, no way around this?
    if explicit_stitching:
        stitching_data = [d[1:] for d in stitching_data]

    # Get whole sample shape. Unique to get rid of batch dimension. Expects tensors
    input_shape = [x.unique() for x in stitching_data[0][0]]

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
            ] = cropped_tile.to(torch.float32)

    return predicted_image


def tta_forward(x: np.ndarray) -> List:
    """
    Augment 8-fold an array.

    The augmentation is performed using all 90 deg rotations and their flipped version,
    as well as the original image flipped.

    Parameters
    ----------
    x : torch.tensor
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


def tta_backward(x_aug: List) -> np.ndarray:
    """
    Invert `tta_forward` and average the 8 images.

    Parameters
    ----------
    x_aug : List
        Stack of 8-fold augmented images.

    Returns
    -------
    np.ndarray
        Average of de-augmented x_aug.
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
