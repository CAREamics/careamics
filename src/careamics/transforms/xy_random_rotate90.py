"""Patch transform applying XY random 90 degrees rotations."""

from typing import Optional, Tuple

import numpy as np

from careamics.transforms.transform import Transform


class XYRandomRotate90(Transform):
    """Applies random 90 degree rotations to the YX axis.

    This transform expects C(Z)YX dimensions.

    Attributes
    ----------
    rng : np.random.Generator
        Random number generator.
    p : float
        Probability of applying the transform.
    seed : Optional[int]
        Random seed.

    Parameters
    ----------
    p : float
        Probability of applying the transform, by default 0.5.
    seed : Optional[int]
        Random seed, by default None.
    """

    def __init__(self, p: float = 0.5, seed: Optional[int] = None):
        """Constructor.

        Parameters
        ----------
        p : float
            Probability of applying the transform, by default 0.5.
        seed : Optional[int]
            Random seed, by default None.
        """
        if p < 0 or p > 1:
            raise ValueError("Probability must be in [0, 1].")

        # probability to apply the transform
        self.p = p

        # numpy random generator
        self.rng = np.random.default_rng(seed=seed)

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
        if self.rng.random() > self.p:
            return patch, target

        # number of rotations
        n_rot = self.rng.integers(1, 4)

        axes = (-2, -1)
        patch_transformed = self._apply(patch, n_rot, axes)
        target_transformed = (
            self._apply(target, n_rot, axes) if target is not None else None
        )

        return patch_transformed, target_transformed

    def _apply(
        self, patch: np.ndarray, n_rot: int, axes: Tuple[int, int]
    ) -> np.ndarray:
        """Apply the transform to the image.

        Parameters
        ----------
        patch : np.ndarray
            Image or image patch, 2D or 3D, shape C(Z)YX.
        n_rot : int
            Number of 90 degree rotations.
        axes : Tuple[int, int]
            Axes along which to rotate the patch.

        Returns
        -------
        np.ndarray
            Transformed patch.
        """
        return np.ascontiguousarray(np.rot90(patch, k=n_rot, axes=axes))
