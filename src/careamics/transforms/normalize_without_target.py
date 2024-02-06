""" module."""
import albumentations as Aug


class NormalizeWithoutTarget(Aug.DualTransform):
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
        return Aug.functional.normalize(image, self.mean, self.std, self.max_pixel_value)

    def apply_to_mask(self, target, **params):
        """Apply the transform to the mask."""
        return Aug.functional.normalize(target, self.mean, self.std, self.max_pixel_value)
