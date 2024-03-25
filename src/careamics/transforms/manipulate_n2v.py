from typing import Literal, Optional, Tuple

import numpy as np
from albumentations import ImageOnlyTransform

from careamics.config.support import SupportedPixelManipulation

from .pixel_manipulation import median_manipulate, uniform_manipulate


class N2VManipulate(ImageOnlyTransform):
    """
    Default augmentation for the N2V model.

    This transform expects (Z)YXC dimensions.

    Parameters
    ----------
    mask_pixel_percentage : float
        Approximate percentage of pixels to be masked.
    roi_size : int
        Size of the ROI the new pixel value is sampled from, by default 11.
    """

    def __init__(
        self,
        roi_size: int = 11,
        masked_pixel_percentage: float = 0.2,
        strategy: Literal[
            "uniform", "median"
        ] = SupportedPixelManipulation.UNIFORM.value,
        remove_center: bool = True,
        struct_mask: Optional[np.ndarray] = None,
    ):
        super().__init__(p=1)
        self.masked_pixel_percentage = masked_pixel_percentage
        self.roi_size = roi_size
        self.strategy = strategy
        self.remove_center = remove_center
        self.struct_mask = struct_mask

    def apply(
            self, patch: np.ndarray, **kwargs: dict
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply the transform to the image.

        Parameters
        ----------
        image : np.ndarray
            Image or image patch, 2D or 3D, shape (y, x, c) or (z, y, x, c).
        """
        masked = np.zeros_like(patch)
        mask = np.zeros_like(patch)
        if self.strategy == SupportedPixelManipulation.UNIFORM:
            # Iterate over the channels to apply manipulation separately
            for c in range(patch.shape[-1]):
                masked[..., c], mask[..., c] = uniform_manipulate(
                    patch=patch[..., c],
                    mask_pixel_percentage=self.masked_pixel_percentage,
                    subpatch_size=self.roi_size,
                    remove_center=self.remove_center,
                    struct_mask_params=self.struct_mask,
                )
        elif self.strategy == SupportedPixelManipulation.MEDIAN:
            # Iterate over the channels to apply manipulation separately
            for c in range(patch.shape[-1]):
                masked[..., c], mask[..., c] = median_manipulate(
                    patch=patch[..., c],
                    mask_pixel_percentage=self.masked_pixel_percentage,
                    subpatch_size=self.roi_size,
                    struct_mask_params=self.struct_mask,
                )
        else:
            raise ValueError(f"Unknown masking strategy ({self.strategy}).")

        return masked, patch, mask

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("roi_size", "masked_pixel_percentage", "strategy", "struct_mask")
