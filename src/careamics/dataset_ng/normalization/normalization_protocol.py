from typing import Protocol

import torch
from numpy.typing import NDArray


class NormalizationProtocol(Protocol):
    """Protocol for normalization strategies."""

    def __call__(
        self, patch: NDArray, target: NDArray | None = None
    ) -> tuple[NDArray, NDArray | None]:
        """Apply the normalization to the patch and target.

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
        ...

    def denormalize(self, patch: torch.Tensor) -> torch.Tensor:
        """Reverse the normalization operation for a batch of patches.

        Parameters
        ----------
        patch : torch.Tensor
            Patch, 2D or 3D, shape BC(Z)YX.
        """
        ...
