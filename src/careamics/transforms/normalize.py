"""Normalization and denormalization transforms for image patches."""

from typing import Optional, Tuple

import numpy as np

from careamics.transforms.transform import Transform


class Normalize(Transform):
    """
    Normalize an image or image patch.

    Normalization is a zero mean and unit variance. This transform expects C(Z)YX
    dimensions.

    Not that an epsilon value of 1e-6 is added to the standard deviation to avoid
    division by zero and that it returns a float32 image.

    Parameters
    ----------
    mean : float
        Mean value.
    std : float
        Standard deviation value.

    Attributes
    ----------
    mean : float
        Mean value.
    std : float
        Standard deviation value.
    """

    def __init__(
        self,
        mean: float,
        std: float,
    ):
        """Constructor.

        Parameters
        ----------
        mean : float
            Mean value.
        std : float
            Standard deviation value.
        """
        self.mean = mean
        self.std = std
        self.eps = 1e-6

    def __call__(
        self, patch: np.ndarray, target: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply the transform to the source patch and the target (optional).

        Parameters
        ----------
        patch : np.ndarray
            Patch, 2D or 3D, shape C(Z)YX.
        target : Optional[np.ndarray], optional
            Target for the patch, by default None.

        Returns
        -------
        Tuple[np.ndarray, Optional[np.ndarray]]
            Transformed patch and target.
        """
        norm_patch = self._apply(patch)
        norm_target = self._apply(target) if target is not None else None

        return norm_patch, norm_target

    def _apply(self, patch: np.ndarray) -> np.ndarray:
        """
        Apply the transform to the image.

        Parameters
        ----------
        patch : np.ndarray
            Image patch, 2D or 3D, shape C(Z)YX.

        Returns
        -------
        np.ndarray
            Normalizedimage patch.
        """
        return ((patch - self.mean) / (self.std + self.eps)).astype(np.float32)


class Denormalize:
    """
    Denormalize an image.

    Denormalization is performed expecting a zero mean and unit variance input. This
    transform expects C(Z)YX dimensions.

    Note that an epsilon value of 1e-6 is added to the standard deviation to avoid
    division by zero during the normalization step, which is taken into account during
    denormalization.

    Parameters
    ----------
    mean : float
        Mean value.
    std : float
        Standard deviation value.

    Attributes
    ----------
    mean : float
        Mean value.
    std : float
        Standard deviation value.
    """

    def __init__(
        self,
        mean: float,
        std: float,
    ):
        """Constructor.

        Parameters
        ----------
        mean : float
            Mean.
        std : float
            Standard deviation.
        """
        self.mean = mean
        self.std = std
        self.eps = 1e-6

    def __call__(self, patch: np.ndarray) -> np.ndarray:
        """Apply the transform to the source patch and the target (optional).

        Parameters
        ----------
        patch : np.ndarray
            Patch, 2D or 3D, shape C(Z)YX.

        Returns
        -------
        np.ndarray
            Transformed patch.
        """
        norm_patch = self._apply(patch)

        return norm_patch

    def _apply(self, patch: np.ndarray) -> np.ndarray:
        """
        Apply the transform to the image.

        Parameters
        ----------
        patch : np.ndarray
            Image patch, 2D or 3D, shape C(Z)YX.

        Returns
        -------
        np.ndarray
            Denormalized image patch.
        """
        return patch * (self.std + self.eps) + self.mean
