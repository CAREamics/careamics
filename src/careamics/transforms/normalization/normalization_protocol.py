"""Normalization protocol."""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class NormalizationProtocol(Protocol):
    """Protocol for normalization strategies."""

    def __call__(
        self,
        patch: np.ndarray,
        target: np.ndarray | None = None,
        **additional_arrays: np.ndarray,
    ) -> tuple[NDArray, NDArray | None, dict[str, NDArray]]:
        """Apply the normalization to the patch and target.

        Parameters
        ----------
        patch : NDArray
            Patch, 2D or 3D, shape C(Z)YX.
        target : NDArray, optional
            Target for the patch, by default None.
        **additional_arrays : NDArray
            Additional arrays that will be transformed identically to `patch` and
            `target`.

        Returns
        -------
        tuple of NDArray
            Transformed patch and target, the target can be returned as `None`.
        """
        ...

    def denormalize(self, patch: np.ndarray) -> NDArray:
        """Reverse the normalization operation for a batch of patches.

        Parameters
        ----------
        patch : NDArray
            Patch, 2D or 3D, shape BC(Z)YX.
        """
        ...
