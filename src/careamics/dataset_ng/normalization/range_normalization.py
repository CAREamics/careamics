import numpy as np
import torch
from numpy.typing import NDArray

from .normalization_protocol import NormalizationProtocol
from .utils import broadcast_stats, reshape_stats


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
        n_channels = patch.shape[0]
        input_mins = broadcast_stats(self.input_mins, n_channels, "input_mins")
        input_maxes = broadcast_stats(self.input_maxes, n_channels, "input_maxes")

        patch = patch.astype(np.float32)
        if target is not None:
            target = target.astype(np.float32)

        min_val = reshape_stats(input_mins, patch.ndim, channel_axis=0)
        max_val = reshape_stats(input_maxes, patch.ndim, channel_axis=0)

        norm_patch = (patch - min_val) / (max_val - min_val + self.eps)

        norm_target = None
        if target is not None:
            if self.target_mins is None or self.target_maxes is None:
                raise ValueError(
                    "Target mins and maxs must be provided if target is not None."
                )
            n_target_ch = target.shape[0]
            t_mins = broadcast_stats(self.target_mins, n_target_ch, "target_mins")
            t_maxes = broadcast_stats(self.target_maxes, n_target_ch, "target_maxes")
            target_mins_arr = reshape_stats(t_mins, target.ndim, channel_axis=0)
            target_maxes_arr = reshape_stats(t_maxes, target.ndim, channel_axis=0)
            norm_target = (target - target_mins_arr) / (
                target_maxes_arr - target_mins_arr + self.eps
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
        n_channels = patch.shape[1]
        input_mins_list = broadcast_stats(self.input_mins, n_channels, "input_mins")
        input_maxes_list = broadcast_stats(self.input_maxes, n_channels, "input_maxes")

        patch = patch.to(dtype=torch.float32)

        mins_arr = reshape_stats(input_mins_list, patch.ndim, channel_axis=1)
        maxes_arr = reshape_stats(input_maxes_list, patch.ndim, channel_axis=1)

        input_mins = torch.from_numpy(mins_arr).to(patch.device)
        input_maxes = torch.from_numpy(maxes_arr).to(patch.device)

        return patch * (input_maxes - input_mins) + input_mins
