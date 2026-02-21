import numpy as np
import torch
from numpy.typing import NDArray

from .normalization_protocol import NormalizationProtocol
from .utils import broadcast_stats, reshape_stats


class MeanStdNormalization(NormalizationProtocol):
    """
    Normalize an image or image patch.

    Normalization is a zero mean and unit variance. This transform expects C(Z)YX
    dimensions.

    Parameters
    ----------
    input_means : list[float]
        Mean values (length 1 for global, multiple values for per channel).
    input_stds : list[float]
        Standard deviation values (length 1 for global,
        multiple values for per channel).
    target_means : list[float] | None, optional
        Target mean values (length 1 for global,
        multiple values for per channel), by default None.
    target_stds : list[float] | None, optional
        Target standard deviation values (length 1 for global,
        multiple values for per channel), by default None.
    """

    def __init__(
        self,
        input_means: list[float],
        input_stds: list[float],
        target_means: list[float] | None = None,
        target_stds: list[float] | None = None,
    ):
        self.input_means = input_means
        self.input_stds = input_stds
        self.target_means = target_means
        self.target_stds = target_stds

        self.eps = 1e-6

    def __call__(
        self,
        patch: np.ndarray,
        target: NDArray | None = None,
    ) -> tuple[NDArray, NDArray | None]:
        """Apply the transform to the source patch and the target (optional).

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
        n_channels = patch.shape[0]
        input_means = broadcast_stats(self.input_means, n_channels, "input_means")
        input_stds = broadcast_stats(self.input_stds, n_channels, "input_stds")

        patch = patch.astype(np.float32)
        if target is not None:
            target = target.astype(np.float32)

        means = reshape_stats(input_means, patch.ndim, channel_axis=0)
        stds = reshape_stats(input_stds, patch.ndim, channel_axis=0)
        norm_patch = self._apply_normalization(patch, means, stds)

        if target is None:
            norm_target = None
        else:
            if self.target_means is None or self.target_stds is None:
                raise ValueError(
                    "Target means and standard deviations must be provided "
                    "if target is not None."
                )
            n_target_channels = target.shape[0]
            t_means = broadcast_stats(
                self.target_means, n_target_channels, "target_means"
            )
            t_stds = broadcast_stats(self.target_stds, n_target_channels, "target_stds")
            target_means = reshape_stats(t_means, target.ndim, channel_axis=0)
            target_stds = reshape_stats(t_stds, target.ndim, channel_axis=0)
            norm_target = self._apply_normalization(target, target_means, target_stds)

        return norm_patch, norm_target

    def _apply_normalization(
        self, patch: NDArray, mean: NDArray, std: NDArray
    ) -> NDArray:
        """
        Apply the transform to the image.

        Parameters
        ----------
        patch : NDArray
            Image patch, 2D or 3D, shape C(Z)YX.
        mean : NDArray
            Mean values.
        std : NDArray
            Standard deviations.

        Returns
        -------
        NDArray
            Normalized image patch.
        """
        return (patch - mean) / (std + self.eps)

    def _apply_denormalization(
        self, array: torch.Tensor, mean: NDArray, std: NDArray
    ) -> torch.Tensor:
        """
        Apply the transform to the image.

        Parameters
        ----------
        array : torch.Tensor
            Image patch, 2D or 3D, shape BC(Z)YX.
        mean : NDArray
            Mean values.
        std : NDArray
            Standard deviations.

        Returns
        -------
        torch.Tensor
            Denormalized image array.
        """
        mean_tensor = torch.from_numpy(mean).to(array.device)
        std_tensor = torch.from_numpy(std).to(array.device)
        return array * (std_tensor + self.eps) + mean_tensor

    def denormalize(self, patch: torch.Tensor) -> torch.Tensor:
        """Reverse the normalization operation for a batch of patches.

        Parameters
        ----------
        patch : torch.Tensor
            Patch, 2D or 3D, shape BC(Z)YX.

        Returns
        -------
        torch.Tensor
            Transformed array.
        """
        n_channels = patch.shape[1]
        input_means = broadcast_stats(self.input_means, n_channels, "input_means")
        input_stds = broadcast_stats(self.input_stds, n_channels, "input_stds")

        patch = patch.to(dtype=torch.float32)

        means = reshape_stats(input_means, patch.ndim, channel_axis=1)
        stds = reshape_stats(input_stds, patch.ndim, channel_axis=1)

        return self._apply_denormalization(patch, means, stds)
