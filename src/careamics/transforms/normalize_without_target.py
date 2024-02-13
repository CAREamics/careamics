"""TODO."""
import albumentations as Aug
import numpy as np


# TODO: interplay validation without mean/std, but extracted from config instantiation
# TODO maybe rename
# TODO what is this doing exactly? it says without target, but a mask is a target...
class NormalizeWithoutTarget(Aug.DualTransform):
    """Normalize the image with a mask.

    # TODO: add more details.

    Parameters
    ----------
    mean : float
        Mean value.
    std : float
        Standard deviation.
    """

    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        max_pixel_value: float = 1,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(always_apply, p)
        self.mean = mean
        self.std = std
        self.max_pixel_value = max_pixel_value

    def apply(self, image: np.ndarray) -> np.ndarray:
        """TODO.

        Parameters
        ----------
        image : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        return Aug.functional.normalize(
            image, self.mean, self.std, self.max_pixel_value
        )

    def apply_to_mask(self, target: np.ndarray) -> np.ndarray:
        """TODO."""
        return Aug.functional.normalize(
            target, self.mean, self.std, self.max_pixel_value
        )
