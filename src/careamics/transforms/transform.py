"""A general parent class for transforms."""

from typing import Optional, Tuple

import numpy as np


class Transform:
    """A general parent class for transforms."""

    def __call__(
        self, patch: np.ndarray, target: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, ...]:
        """Apply the transform to the input data.

        Parameters
        ----------
        patch : np.ndarray
            The input data to transform.
        target : Optional[np.ndarray], optional
            The target data to transform, by default None

        Returns
        -------
        Tuple[np.ndarray, ...]
            The output of the transformations.

        Raises
        ------
        NotImplementedError
            This method should be implemented in the child class.
        """
        raise NotImplementedError
