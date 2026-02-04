from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from .normalization_protocol import NormalizationProtocol


class NoNormalization(NormalizationProtocol):
    """No normalization transform. Returns the patch as is.

    Parameters
    ----------
    **kwargs : Any
        Additional keyword arguments.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the no normalization transform.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments.
        """
        pass

    def __call__(
        self, patch: NDArray, target: NDArray | None = None
    ) -> tuple[NDArray, NDArray | None]:
        """Apply no normalization to the patch and target.

        Parameters
        ----------
        patch : NDArray
            Patch, 2D or 3D, shape C(Z)YX.
        target : NDArray, optional
            Target for the patch, by default None.

        Returns
        -------
        tuple of NDArray
            Transformed patch and target, the target can be returned as `None`.
        """
        patch = patch.astype(np.float32)
        if target is not None:
            target = target.astype(np.float32)
        return patch, target

    def denormalize(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Reverse the normalization operation for a batch of patches.

        Parameters
        ----------
        patch : torch.Tensor
            Patch, 2D or 3D, shape BC(Z)YX.

        Returns
        -------
        torch.Tensor
            Denormalized patch.
        """
        return patch.to(dtype=torch.float32)
