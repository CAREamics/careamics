from typing import Optional, Tuple, Union

import numpy as np
from albumentations import ImageOnlyTransform

from careamics.config.support import SupportedPixelManipulation

from .pixel_manipulation import median_manipulate, uniform_manipulate


# TODO add median vs random replace
class N2VManipulateUniform(ImageOnlyTransform):
    """
    Default augmentation for the N2V model.

    This transform expects S(Z)YXC dimensions.

    # TODO add more details, in paritcular what happens to channels and Z in the masking

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
        strategy: Union[
            str, SupportedPixelManipulation
        ] = SupportedPixelManipulation.UNIFORM,
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
            Image or image patch, 2D or 3D, shape (y, x, c) or (z, y, x, c).
        """
        if self.strategy == SupportedPixelManipulation.UNIFORM:
            masked, mask = uniform_manipulate(
                patch=patch,
                mask_pixel_percentage=self.masked_pixel_percentage,
                subpatch_size=self.roi_size,
                struct_mask_params=self.struct_mask,  # TODO add remove center param
            )
        else:
            raise ValueError(f"Strategy {self.strategy} not supported.")

        return manipulated, patch, mask

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("roi_size", "masked_pixel_percentage", "strategy", "struct_mask")


class N2VManipulateMedian(ImageOnlyTransform):
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
