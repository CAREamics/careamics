"""Patch transform randomly swapping input and target arrays."""

import numpy as np
from numpy.typing import NDArray

from careamics.dataset.augmentation.transform import Transform


class SwapInputTarget(Transform):
    """Randomly swap the input and target patches.

    Parameters
    ----------
    p : float, default=1.0
        Probability of swapping input and target.
    seed : int | None, default=None
        Seed for the random number generator.
    """

    def __init__(self, p: float = 0.5, seed: int | None = None) -> None:
        """Initialize the transform.

        Parameters
        ----------
        p : float, default=1.0
            Probability of swapping input and target.
        seed : int | None, default=None
            Seed for the random number generator.
        """
        if p < 0 or p > 1:
            raise ValueError("Probability must be in [0, 1].")
        self.p = p
        self.rng = np.random.default_rng(seed=seed)

    def __call__(
        self,
        patch: NDArray,
        target: NDArray | None = None,
        **additional_arrays: NDArray,
    ) -> tuple[NDArray, NDArray | None, dict[str, NDArray]]:
        """Potentially exchange the input and target patches.

        Parameters
        ----------
        patch : numpy.ndarray
            Input patch.
        target : numpy.ndarray | None, default=None
            Target patch.
        **additional_arrays : numpy.ndarray
            Additional arrays, which are returned unchanged.

        Returns
        -------
        numpy.ndarray
            Input.
        numpy.ndarray | None
            Target.
        dict[str, numpy.ndarray]]
            Unchanged additional arrays.
        """
        if target is None or self.rng.random() > self.p:
            return patch, target, additional_arrays
        return target, patch, additional_arrays
