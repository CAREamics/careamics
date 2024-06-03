"""XY flip transform."""

from typing import Literal, Optional, Tuple

import numpy as np

from careamics.transforms.transform import Transform


class XorYFlip(Transform):
    """Flip image along a single axis.

    This transform expects C(Z)YX dimensions.

    Attributes
    ----------
    axis : Literal[-2, -1]
        Axis to be flipped.
    rng : np.random.Generator
        Random number generator.
    p : float
        Probability of applying the transform.
    seed : Optional[int]
        Random seed.

    Parameters
    ----------
    axis : Literal[-2, -1]
        Axis to flip.
    p : float, optional
        Probability of applying the transform, by default 0.5.
    seed : Optional[int], optional
        Random seed, by default None.
    """

    def __init__(
        self, axis: Literal[-2, -1], p: float = 0.5, seed: Optional[int] = None
    ) -> None:
        """Constructor.

        Parameters
        ----------
        axis : Literal[-2, -1]
            Axis to flip.
        p : float
            Probability of applying the transform, by default 0.5.
        seed : Optional[int], optional
            Random seed, by default None.
        """
        if p < 0 or p > 1:
            raise ValueError("Probability must be in [0, 1].")

        # axis to flip
        self.axis = axis

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

        patch_transformed = self._apply(patch, self.axis)
        target_transformed = (
            self._apply(target, self.axis) if target is not None else None
        )

        return patch_transformed, target_transformed

    def _apply(self, patch: np.ndarray, axis: int) -> np.ndarray:
        """Apply the transform to the image.

        Parameters
        ----------
        patch : np.ndarray
            Image patch, 2D or 3D, shape C(Z)YX.
        axis : int
            Axis to flip.

        Returns
        -------
        np.ndarray
            Flipped image patch.
        """
        return np.ascontiguousarray(np.flip(patch, axis=axis))
