"""No normalization transform."""

from typing import Any

from numpy.typing import NDArray

from ..transform import Transform
from .normalization_protocol import NormalizationProtocol


class NoNormalization(NormalizationProtocol, Transform):
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
        self,
        patch: NDArray,
        target: NDArray | None = None,
        **additional_arrays: NDArray,
    ) -> tuple[NDArray, NDArray | None, dict[str, NDArray]]:
        """Apply no normalization to the patch and target.

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
        return patch, target, additional_arrays

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
