from typing import Any, Tuple

import numpy as np
from albumentations import DualTransform


class Denormalize(DualTransform):
    """Denormalize an image or image patch.

    This transform expects (Z)YXC dimensions.
    """

    def __init__(
        self,
        mean: Tuple[float],
        std: Tuple[float],
        max_pixel_value: int = 1,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(always_apply=always_apply, p=p)

        self.mean = np.array(mean)
        self.std = np.array(std)
        self.max_pixel_value = max_pixel_value

    def apply(self, patch: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply the transform to the image.

        Parameters
        ----------
        patch : np.ndarray
            Image or image patch, 2D or 3D, shape (y, x, c) or (z, y, x, c).
        """
        return (patch * self.std + self.mean) * self.max_pixel_value

    def apply_to_mask(self, mask: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply the transform to the mask.

        Parameters
        ----------
        mask : np.ndarray
            Mask or mask patch, 2D or 3D, shape (y, x, c) or (z, y, x, c).
        """
        return mask
