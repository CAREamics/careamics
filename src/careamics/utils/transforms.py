"""Augmentation module."""

import albumentations as A
import numpy as np

from ..manipulation import median_manipulate, uniform_manipulate


class NormalizeWithoutTarget(A.DualTransform):
    """
    Normalize the image with a mask.

    Parameters
    ----------
    mean : float
        Mean value.
    std : float
        Standard deviation.
    """

    def __init__(
        self,
        mean: float,
        std: float,
        max_pixel_value=1,
        always_apply=False,
        p=1.0,
    ):
        super().__init__(always_apply, p)
        self.mean = mean
        self.std = std
        self.max_pixel_value = max_pixel_value

    def apply(self, image, **params):
        """Apply the transform to the mask."""
        return A.functional.normalize(image, self.mean, self.std, self.max_pixel_value)

    def apply_to_mask(self, target, **params):
        """Apply the transform to the mask."""
        return A.functional.normalize(target, self.mean, self.std, self.max_pixel_value)


class N2VManipulateUniform(A.ImageOnlyTransform):
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
        remove_center: bool = False,
        struct_mask: np.ndarray = None,
    ):
        super().__init__(p=1)
        self.masked_pixel_percentage = masked_pixel_percentage
        self.roi_size = roi_size
        self.remove_center = remove_center
        self.struct_mask = struct_mask

    def apply(self, image, **params):
        """Apply the transform to the image.

        Parameters
        ----------
        image : np.ndarray
            Image or image patch, 2D or 3D, shape (c, y, x) or (c, z, y, x).
        """
        masked, original, mask = uniform_manipulate(
            image,
            self.masked_pixel_percentage,
            self.roi_size,
            self.remove_center,
            self.struct_mask,
        )
        return masked, original, mask


class N2VManipulateMedian(A.ImageOnlyTransform):
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
        masked, original, mask = median_manipulate(
            image, self.masked_pixel_percentage, self.roi_size, self.struct_mask
        )
        return masked, original, mask
