import numpy as np
import torch
from numpy.typing import NDArray

from .normalization_protocol import NormalizationProtocol


def _reshape_stats(stats: list[float], ndim: int, channel_axis: int = 0) -> NDArray:
    """Reshape stats to match the number of dimensions of the input image.

    Parameters
    ----------
    stats : list of float
        List of per-channel statistic values.
    ndim : int
        Number of dimensions in the image.
    channel_axis : int, optional
        Position of the channel axis in the image.

    Returns
    -------
    NDArray
        Stats array reshaped, with the channel dimension
        at the specified axis and singleton dimensions elsewhere.
    """
    shape = [1] * ndim
    shape[channel_axis] = len(stats)
    return np.array(stats, dtype=np.float32).reshape(shape)


class MeanStdNormalization(NormalizationProtocol):
    """
    Normalize an image or image patch.

    Normalization is a zero mean and unit variance. This transform expects C(Z)YX
    dimensions.

    Note that an epsilon value of 1e-6 is added to the standard deviation to avoid
    division by zero and that it returns a float32 image.

    Parameters
    ----------
    input_means : list of float
        Mean value per channel.
    input_stds : list of float
        Standard deviation value per channel.
    target_means : list of float, optional
        Target mean value per channel, by default None.
    target_stds : list of float, optional
        Target standard deviation value per channel, by default None.

    Attributes
    ----------
    input_means : list of float
        Mean value per channel.
    input_stds : list of float
        Standard deviation value per channel.
    target_means : list of float, optional
        Target mean value per channel, by default None.
    target_stds : list of float, optional
        Target standard deviation value per channel, by default None.
    """

    def __init__(
        self,
        input_means: list[float],
        input_stds: list[float],
        target_means: list[float] | None = None,
        target_stds: list[float] | None = None,
    ):
        """Constructor.

        Parameters
        ----------
        input_means : list of float
            Mean value per channel.
        input_stds : list of float
            Standard deviation value per channel.
        target_means : list of float, optional
            Target mean value per channel, by default None.
        target_stds : list of float, optional
            Target standard deviation value per channel, by default None.
        """
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
        if len(self.input_means) != patch.shape[0]:
            raise ValueError(
                f"Number of means (got a list of size {len(self.input_means)}) and "
                f"number of channels (got shape {patch.shape} for C(Z)YX) do not match."
            )

        patch = patch.astype(np.float32)
        if target is not None:
            target = target.astype(np.float32)

        # reshape mean and std and apply the normalization to the patch
        means = _reshape_stats(self.input_means, patch.ndim, channel_axis=0)
        stds = _reshape_stats(self.input_stds, patch.ndim, channel_axis=0)
        norm_patch = self._apply_normalization(patch, means, stds)

        # same for the target patch
        if target is None:
            norm_target = None
        else:
            if not self.target_means or not self.target_stds:
                raise ValueError(
                    "Target means and standard deviations must be provided "
                    "if target is not None."
                )
            target_means = _reshape_stats(
                self.target_means, target.ndim, channel_axis=0
            )
            target_stds = _reshape_stats(self.target_stds, target.ndim, channel_axis=0)
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
        if len(self.input_means) != patch.shape[1]:
            raise ValueError(
                f"Number of means (got a list of size {len(self.input_means)}) and "
                f"number of channels (got shape {patch.shape} for BC(Z)YX) do not "
                f"match."
            )

        patch = patch.to(dtype=torch.float32)

        means = _reshape_stats(self.input_means, patch.ndim, channel_axis=1)
        stds = _reshape_stats(self.input_stds, patch.ndim, channel_axis=1)

        denorm_array = self._apply_denormalization(patch, means, stds)

        return denorm_array
