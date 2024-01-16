"""Augmentation module."""
from typing import Tuple

import albumentations as A
import numpy as np

from ..manipulation import default_manipulate


class ManipulateN2V(A.ImageOnlyTransform):
    """
    Default augmentation for the N2V model.

    Parameters
    ----------
    mask_pixel_percentage : floar
        Approximate percentage of pixels to be masked.
    roi_size : int
        Size of the ROI the new pixel value is sampled from, by default 11.
    """

    def __init__(
        self,
        masked_pixel_percentage: float = 0.2,
        roi_size: int = 11,
        struct_mask: np.ndarray = None,
    ):
        super().__init__(p=1)
        self.masked_pixel_percentage = masked_pixel_percentage
        self.roi_size = roi_size
        self.struct_mask = struct_mask

    def apply(self, image, **params):
        """Apply the transform to the image.

        Parameters
        ----------
        image : np.ndarray
            Image or image patch, 2D or 3D, shape (c, y, x) or (c, z, y, x).
        """
        masked, original, mask = default_manipulate(
            image, self.masked_pixel_percentage, self.roi_size, self.struct_mask
        )
        return masked, original, mask


def _flip_and_rotate(
    image: np.ndarray, rotate_state: int, flip_state: int
) -> np.ndarray:
    """
    Apply the given number of 90 degrees rotations and flip to an array.

    Parameters
    ----------
    image : np.ndarray
        Array containing single image or patch, 2D or 3D.
    rotate_state : int
        Number of 90 degree rotations to apply.
    flip_state : int
        0 or 1, whether to flip the array or not.

    Returns
    -------
    np.ndarray
        Flipped and rotated array.
    """
    rotated = np.rot90(image, k=rotate_state, axes=(-2, -1))
    flipped = np.flip(rotated, axis=-1) if flip_state == 1 else rotated
    return flipped.copy()


def augment_batch(
    patch: np.ndarray,
    original_image: np.ndarray,
    mask: np.ndarray,
    seed: int = 42,
) -> Tuple[np.ndarray, ...]:
    """
    Apply augmentation function to patches and masks.

    Parameters
    ----------
    patch : np.ndarray
        Array containing single image or patch, 2D or 3D with masked pixels.
    original_image : np.ndarray
        Array containing original image or patch, 2D or 3D.
    mask : np.ndarray
        Array containing only masked pixels, 2D or 3D.
    seed : int, optional
        Seed for random number generator, controls the rotation and falipping.

    Returns
    -------
    Tuple[np.ndarray, ...]
        Tuple of augmented arrays.
    """
    rng = np.random.default_rng(seed=seed)
    rotate_state = rng.integers(0, 4)
    flip_state = rng.integers(0, 2)
    return (
        _flip_and_rotate(patch, rotate_state, flip_state),
        _flip_and_rotate(original_image, rotate_state, flip_state),
        _flip_and_rotate(mask, rotate_state, flip_state),
    )
