import numpy as np
import torch
from numpy.typing import NDArray

from .mean_std_normalization import _reshape_stats
from .normalization_protocol import NormalizationProtocol


class RangeNormalization(NormalizationProtocol):
    """Normalize an image or image patch to [0, 1] range.

    This transform expects C(Z)YX dimensions for normalization and BC(Z)YX
    for denormalization. Returns float32 arrays.

    Parameters
    ----------
    input_mins : list of float
        Minimum value per channel.
    input_maxes : list of float
        Maximum value per channel.
    target_mins : list of float, optional
        Target minimum value per channel.
    target_maxes : list of float, optional
        Target maximum value per channel.

    Attributes
    ----------
    input_mins : list of float
        Minimum value per channel.
    input_maxes : list of float
        Maximum value per channel.
    target_mins : list of float, optional
        Target minimum value per channel.
    target_maxes : list of float, optional
        Target maximum value per channel.
    eps : float
        Small value to avoid division by zero.
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
        self.eps = 1e-8

    def __call__(
        self, patch: NDArray, target: NDArray | None = None
    ) -> tuple[NDArray, NDArray | None]:
        """Apply range normalization to patch and optional target.

        Parameters
        ----------
        patch : NDArray
            Patch with shape C(Z)YX.
        target : NDArray, optional
            Target patch with shape C(Z)YX.

        Returns
        -------
        tuple of NDArray
            Normalized patch and target (target can be None).
        """
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

        patch = patch.astype(np.float32)
        if target is not None:
            target = target.astype(np.float32)

        min_val = _reshape_stats(self.input_mins, patch.ndim, channel_axis=0)
        max_val = _reshape_stats(self.input_maxes, patch.ndim, channel_axis=0)

        norm_patch = (patch - min_val) / (max_val - min_val + self.eps)

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
            target_mins = _reshape_stats(self.target_mins, target.ndim, channel_axis=0)
            target_maxes = _reshape_stats(
                self.target_maxes, target.ndim, channel_axis=0
            )
            norm_target = (target - target_mins) / (
                target_maxes - target_mins + self.eps
            )

        return norm_patch, norm_target

    def denormalize(self, patch: torch.Tensor) -> torch.Tensor:
        """Reverse the normalization operation for a batch of patches.

        Parameters
        ----------
        patch : torch.Tensor
            Normalized patch with shape BC(Z)YX.

        Returns
        -------
        torch.Tensor
            Denormalized patch.
        """
        if len(self.input_mins) != patch.shape[1]:
            raise ValueError(
                f"Number of mins ({len(self.input_mins)}) and number of "
                f"channels ({patch.shape[1]} in {patch.shape} for BC(Z)YX) "
                "do not match."
            )

        patch = patch.to(dtype=torch.float32)

        input_mins = _reshape_stats(self.input_mins, patch.ndim, channel_axis=1)
        input_maxes = _reshape_stats(self.input_maxes, patch.ndim, channel_axis=1)

        input_mins = torch.from_numpy(input_mins).to(patch.device)
        input_maxes = torch.from_numpy(input_maxes).to(patch.device)

        return patch * (input_maxes - input_mins) + input_mins
