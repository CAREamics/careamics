import numpy as np
from numpy.typing import NDArray

from .normalization_protocol import NormalizationProtocol
from .standardization import _reshape_stats


class RangeNormalization(NormalizationProtocol):
    """Normalize an image or image patch.

    Normalization is a range normalization. This transform expects C(Z)YX
    dimensions. Normalizes to [0, 1].
    """

    def __init__(
        self,
        input_mins: list[float],
        input_maxes: list[float],
        target_mins: list[float] | None = None,
        target_maxes: list[float] | None = None,
    ):
        self.input_mins = input_mins
        self.input_maxes = input_maxes
        self.target_mins = target_mins
        self.target_maxes = target_maxes

    def __call__(
        self, patch: NDArray, target: NDArray | None = None
    ) -> tuple[NDArray, NDArray | None]:
        if len(self.input_mins) != patch.shape[0]:
            raise ValueError(
                f"Number of mins ({len(self.input_mins)}) and number of "
                f"channels ({patch.shape[0]} in {patch.shape}) do not match."
            )

        if len(self.input_maxes) != patch.shape[0]:
            raise ValueError(
                f"Number of max values ({len(self.input_maxes)}) and number of "
                f"channels ({patch.shape[0]} in {patch.shape}) do not match."
            )

        min_val = _reshape_stats(self.input_mins, patch.ndim)
        max_val = _reshape_stats(self.input_maxes, patch.ndim)
        min_val = min_val.astype(patch.dtype)
        max_val = max_val.astype(patch.dtype)

        norm_patch = (patch - min_val) / (max_val - min_val)

        norm_target = None
        if target is not None:
            if self.target_mins is None or self.target_maxes is None:
                raise ValueError(
                    "Target mins and maxs must be provided if target is not None."
                )
            if len(self.target_mins) != target.shape[0]:
                raise ValueError(
                    f"Number of target mins ({len(self.target_mins)}) and number of "
                    f"channels ({target.shape[0]} in {target.shape}) do not match."
                )

            if len(self.target_maxes) != target.shape[0]:
                raise ValueError(
                    f"Number of target maxes ({len(self.target_maxes)}) and number "
                    f"of channels ({target.shape[0]} in {target.shape}) do not match."
                )
            target_mins = _reshape_stats(self.target_mins, target.ndim)
            target_maxes = _reshape_stats(self.target_maxes, target.ndim)
            norm_target = (target - target_mins) / (target_maxes - target_mins)

        return norm_patch, norm_target

    def denormalize(self, patch: NDArray) -> NDArray:
        """Reverse the normalization operation.

        Expects BCZYX input (batch dimension present).

        Parameters
        ----------
        patch : NDArray
            Normalized patch with shape BCZYX.

        Returns
        -------
        NDArray
            Denormalized patch.
        """
        input_mins = _reshape_stats(self.input_mins, patch.ndim)
        input_maxes = _reshape_stats(self.input_maxes, patch.ndim)
        # Swap axes to align stats with channel axis (axis 1 in BCZYX)
        input_mins = np.swapaxes(input_mins, 0, 1)
        input_maxes = np.swapaxes(input_maxes, 0, 1)
        input_mins = input_mins.astype(patch.dtype)
        input_maxes = input_maxes.astype(patch.dtype)
        return patch * (input_maxes - input_mins) + input_mins
