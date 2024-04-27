from typing import Any

import numpy as np
from albumentations import DualTransform


class Normalize(DualTransform):
    """
    Normalize an image or image patch.

    Normalization is a zero mean and unit variance. This transform expects (Z)YXC
    dimensions.

    Not that an epsilon value of 1e-6 is added to the standard deviation to avoid
    division by zero and that it returns a float32 image.

    Attributes
    ----------
    mean : float
        Mean value.
    std : float
        Standard deviation value.
    eps : float
        Epsilon value to avoid division by zero.
    """

    def __init__(
        self,
        mean: float,
        std: float,
    ):
        super().__init__(always_apply=True, p=1)

        self.mean = mean
        self.std = std
        self.eps = 1e-6

    def apply(self, patch: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Apply the transform to the image.

        Parameters
        ----------
        patch : np.ndarray
            Image or image patch, 2D or 3D, shape (y, x, c) or (z, y, x, c).

        Returns
        -------
        np.ndarray
            Normalized image or image patch.
        """
        return ((patch - self.mean) / (self.std + self.eps)).astype(np.float32)

    def apply_to_mask(self, mask: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Apply the transform to the mask.

        The mask is returned as is.

        Parameters
        ----------
        mask : np.ndarray
            Mask or mask patch, 2D or 3D, shape (y, x, c) or (z, y, x, c).
        """
        return mask


class Denormalize(DualTransform):
    """
    Denormalize an image or image patch.

    Denormalization is performed expecting a zero mean and unit variance input. This
    transform expects (Z)YXC dimensions.

    Not that an epsilon value of 1e-6 is added to the standard deviation to avoid
    division by zero during the normalization step, which is taken into account during
    denormalization.

    Attributes
    ----------
    mean : float
        Mean value.
    std : float
        Standard deviation value.
    eps : float
        Epsilon value to avoid division by zero.
    """

    def __init__(
        self,
        mean: float,
        std: float,
    ):
        super().__init__(always_apply=True, p=1)

        self.mean = mean
        self.std = std
        self.eps = 1e-6

    def apply(self, patch: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Apply the transform to the image.

        Parameters
        ----------
        patch : np.ndarray
            Image or image patch, 2D or 3D, shape (y, x, c) or (z, y, x, c).
        """
        return patch * (self.std + self.eps) + self.mean
