from typing import Any

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
        return patch, target

    def denormalize(self, patch: NDArray) -> NDArray:
        """
        Reverse the normalization operation for a batch of patches.

        Parameters
        ----------
        patch : NDArray
            Patch, 2D or 3D, shape BC(Z)YX.

        Returns
        -------
        NDArray
            Denormalized patch.
        """
        return patch
