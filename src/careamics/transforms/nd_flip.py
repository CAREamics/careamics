from typing import Optional, Tuple

import numpy as np

from careamics.transforms.transform import Transform


class NDFlip(Transform):
    """Flip ND arrays on a single axis.

    This transform ignores singleton axes and randomly flips one of the other
    last two axes.

    This transform expects C(Z)YX dimensions.
    """

    def __init__(self, seed: Optional[int] = None):
        """Constructor.

        Parameters
        ----------
        seed : Optional[int], optional
            Random seed, by default None
        """
        # "flippable" axes
        self.axis_indices = [-2, -1]

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
            Target for the patch, by default None

        Returns
        -------
        Tuple[np.ndarray, Optional[np.ndarray]]
            Transformed patch and target.
        """
        # choose an axis to flip
        axis = self.rng.choice(self.axis_indices)

        patch_transformed = self._apply(patch, axis)
        target_transformed = self._apply(target, axis) if target is not None else None

        return patch_transformed, target_transformed

    def _apply(self, patch: np.ndarray, axis: int) -> np.ndarray:
        """Apply the transform to the image.

        Parameters
        ----------
        patch : np.ndarray
            Image or image patch, 2D or 3D, shape C(Z)YX.
        axis : int
            Axis to flip.
        """
        # TODO why ascontiguousarray?
        return np.ascontiguousarray(np.flip(patch, axis=axis))
