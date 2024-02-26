from typing import Optional, Union

import numpy as np
from albumentations import ImageOnlyTransform

from careamics.config.support import SupportedPixelManipulation

from .pixel_manipulation import uniform_manipulate


# TODO add median vs random replace
class ManipulateN2V(ImageOnlyTransform):
    """
    Default augmentation for the N2V model.

    # TODO add more details

    Parameters
    ----------
    mask_pixel_percentage : floar
        Approximate percentage of pixels to be masked.
    roi_size : int
        Size of the ROI the new pixel value is sampled from, by default 11.
    """

    def __init__(
        self,
        roi_size: int = 11,
        masked_pixel_percentage: float = 0.2,
        strategy: Union[str, SupportedPixelManipulation]
          = SupportedPixelManipulation.UNIFORM,
        struct_mask: Optional[np.ndarray] = None,
    ):
        super().__init__(p=1)
        self.masked_pixel_percentage = masked_pixel_percentage
        self.roi_size = roi_size
        self.strategy = strategy
        self.struct_mask = struct_mask

    def apply(self, patch: np.ndarray, **kwargs: dict) -> np.ndarray:
        """Apply the transform to the image.

        Parameters
        ----------
        image : np.ndarray
            Image or image patch, 2D or 3D, shape (c, y, x) or (c, z, y, x).
        """
        if self.strategy == SupportedPixelManipulation.UNIFORM:
            masked, mask = uniform_manipulate(
                patch, self.roi_size, self.masked_pixel_percentage, self.struct_mask
            )
        else:
            raise ValueError(f"Strategy {self.strategy} not supported.")

        return masked, patch, mask