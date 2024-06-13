"""Normalization and denormalization transforms for image patches."""

from typing import Optional, Union

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
    image_means : List[float]
        Mean value per channel.
    image_stds : List[float]
        Standard deviation value per channel.
    target_means : Optional[List[float]], optional
        Target mean value per channel, by default None.
    target_stds : Optional[List[float]], optional
        Target standard deviation value per channel, by default None.

    Attributes
    ----------
    image_means : list[float]
        Mean value per channel.
    image_stds : list[float]
        Standard deviation value per channel.
    target_means : Optional[list[float]], optional
        Target mean value per channel, by default None.
    target_stds : Optional[list[float]], optional
        Target standard deviation value per channel, by default None.
    """

    def __init__(
        self,
        image_means: list[float],
        image_stds: list[float],
        target_means: Optional[list[float]] = None,
        target_stds: Optional[list[float]] = None,
    ):
        """Constructor.

        Parameters
        ----------
        image_means : list[float]
            Mean value per channel.
        image_stds : list[float]
            Standard deviation value per channel.
        target_means : Optional[list[float]], optional
            Target mean value per channel, by default None.
        target_stds : Optional[list[float]], optional
            Target standard deviation value per channel, by default None.
        """
        self.image_means = image_means
        self.image_stds = image_stds
        self.target_means = target_means
        self.target_stds = target_stds

        self.eps = 1e-6

    def __call__(
        self, patch: np.ndarray, target: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
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
        assert (
            len(self.image_means) == patch.shape[0]
        ), "Number of means and number of channels do not match."
        means = np.array(self.image_means)[(..., *[np.newaxis] * (patch.ndim - 1))]
        stds = np.array(self.image_stds)[(..., *[np.newaxis] * (patch.ndim - 1))]
        norm_patch = self._apply(patch, means, stds)

        if target is not None:
            target_means = np.array(self.target_means)[
                (..., *[np.newaxis] * (target.ndim - 1))
            ]
            target_stds = np.array(self.target_stds)[
                (..., *[np.newaxis] * (target.ndim - 1))
            ]
            norm_target = self._apply(target, means, stds)
            norm_target = self._apply(target, target_means, target_stds)
        else:
            norm_target = None

        return norm_patch, norm_target

    def _apply(
        self, patch: np.ndarray, mean: np.ndarray, std: np.ndarray
    ) -> np.ndarray:
        """
        Apply the transform to the image.

        Parameters
        ----------
        patch : np.ndarray
            Image patch, 2D or 3D, shape C(Z)YX.
        mean : float
            Mean value.
        std : float
            Standard deviation.

        Returns
        -------
        np.ndarray
            Normalized image patch.
        """
        return ((patch - mean) / (std + self.eps)).astype(np.float32)


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
    image_means : Tuple[float]
        Mean value per channel.
    image_stds : Tuple[float]
        Standard deviation value per channel.

    """

    def __init__(
        self,
        image_means: Union[list[float], tuple[float]],
        image_stds: Union[list[float], tuple[float]],
    ):
        """Constructor.

        Parameters
        ----------
        image_means : Union[list[float], tuple[float]]
            Mean value per channel.
        image_stds : Union[list[float], tuple[float]]
            Standard deviation value per channel.
        """
        self.image_means = image_means
        self.image_stds = image_stds

        self.eps = 1e-6

    def __call__(self, patch: np.ndarray) -> np.ndarray:
        """Reverse the normalization operation for a batch of patches.

        Parameters
        ----------
        patch : np.ndarray
            Patch, 2D or 3D, shape BC(Z)YX.

        Returns
        -------
        Tuple[np.ndarray]
            Transformed array.
        """
        assert (
            len(self.image_means) == patch.shape[1]
        ), "Number of means and number of channels do not match."

        means = np.array(self.image_means)[(..., *[np.newaxis] * (patch.ndim - 1))]
        stds = np.array(self.image_stds)[(..., *[np.newaxis] * (patch.ndim - 1))]

        denorm_array = self._apply(
            patch, np.swapaxes(means, 0, 1), np.swapaxes(stds, 0, 1)
        )

        return denorm_array.astype(np.float32)

    def _apply(
        self, array: np.ndarray, mean: np.ndarray, std: np.ndarray
    ) -> np.ndarray:
        """
        Apply the transform to the image.

        Parameters
        ----------
        array : np.ndarray
            Image patch, 2D or 3D, shape C(Z)YX.
        mean : float
            Mean value.
        std : float
            Standard deviation.

        Returns
        -------
        np.ndarray
            Denormalized image array.
        """
        return array * (std + self.eps) + mean
