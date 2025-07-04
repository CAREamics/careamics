"""Normalization and denormalization transforms for image patches using quantiles."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from careamics.transforms.transform import Transform

from .normalization_protocol import NormalizationProtocol


class QuantileNormalization(NormalizationProtocol, Transform):
    """
    Normalize an image or image patch using quantiles. Input is normalized patch wise.

    Parameters
    ----------
    lower : float, default=0.01
        Lower quantile to normalize to.
    upper : float, default=0.99
        Upper quantile to normalize to.
    target_lower : float | None, default=None
        Lower quantile to normalize target to.
    target_upper : float | None, default=None
        Upper quantile to normalize target to.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    tuple[NDArray, NDArray | None, dict[str, NDArray]]
        Normalized patch, normalized target, and additional arrays.
    """

    def __init__(
        self,
        lower: float = 0.01,
        upper: float = 0.99,
        target_lower: float | None = None,
        target_upper: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the quantile normalization transform.

        Parameters
        ----------
        lower : float, default=0.01
            Lower quantile to normalize to.
        upper : float, default=0.99
            Upper quantile to normalize to.
        target_lower : float | None, default=None
            Lower quantile to normalize target to.
        target_upper : float | None, default=None
            Upper quantile to normalize target to.
        **kwargs : Any
            Additional keyword arguments.
        """
        self.lower = lower
        self.upper = upper
        self.target_lower = target_lower
        self.target_upper = target_upper

        self.eps = 1e-6

    def __call__(
        self,
        patch: NDArray,
        target: NDArray | None = None,
        **additional_arrays: NDArray,
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
        tuple[NDArray, NDArray | None, dict[str, NDArray]]
            Normalized patch, normalized target, and additional arrays.
        """
        if len(additional_arrays) != 0:
            raise NotImplementedError(
                "Transforming additional arrays is currently "
                "not supported for `QuantileNormalization`."
            )

        norm_patch = self._apply_normalization(patch, self.lower, self.upper)

        if target is None:
            norm_target = None
        else:
            norm_target = self._apply_normalization(target, self.lower, self.upper)

        return norm_patch, norm_target, additional_arrays

    # TODO: figure out how to handle channel axis
    def _apply_normalization(
        self, patch: NDArray, lower: float, upper: float
    ) -> NDArray:
        """Apply the normalization to the patch.

        Parameters
        ----------
        patch : NDArray
            Patch, 2D or 3D, shape BC(Z)YX.
        lower : float
            Lower quantile to normalize to.
        upper : float
            Upper quantile to normalize to.

        Returns
        -------
        NDArray
            Normalized patch.
        """
        # Compute percentiles per patch, per channel
        axes = tuple(i for i in range(patch.ndim) if i != 0)
        pmin = np.percentile(patch, lower * 100, axis=axes, keepdims=True)
        pmax = np.percentile(patch, upper * 100, axis=axes, keepdims=True)
        return ((patch - pmin) / (pmax - pmin + self.eps)).astype(np.float32)

    def denormalize(self, patch: NDArray) -> NDArray:
        """Reverse the normalization operation for a batch of patches.

        Parameters
        ----------
        patch : NDArray
            Patch, 2D or 3D, shape BC(Z)YX.

        Returns
        -------
        NDArray
            Denormalized patch.
        """
        return self._apply_denormalization(patch, self.lower, self.upper)

    def _apply_denormalization(
        self, patch: NDArray, min_value: float, max_value: float
    ) -> NDArray:
        """Apply the denormalization to the patch.

        Parameters
        ----------
        patch : NDArray
            Patch, 2D or 3D, shape BC(Z)YX.
        min_value : float
            Minimum value.
        max_value : float
            Maximum value.

        Returns
        -------
        NDArray
            Denormalized patch.
        """
        min_value = np.percentile(min_value, self.lower * 100, axis=0, keepdims=True)
        return (patch * (max_value - min_value + self.eps) + min_value).astype(
            np.float32
        )
