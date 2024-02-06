from albumentations import ImageOnlyTransform
import numpy as np

from .pixel_manipulation import default_manipulate


class ManipulateN2V(ImageOnlyTransform):
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

