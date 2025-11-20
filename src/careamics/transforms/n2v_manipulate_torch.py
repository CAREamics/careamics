"""N2V manipulation transform for PyTorch."""

import platform
from typing import Any

import torch

from careamics.config.support import SupportedPixelManipulation, SupportedStructAxis
from careamics.config.transformations import N2VManipulateConfig

from .pixel_manipulation_torch import (
    median_manipulate_torch,
    uniform_manipulate_torch,
)
from .struct_mask_parameters import StructMaskParameters


class N2VManipulateTorch:
    """
    N2V manipulation with asymmetric masking and channel dropout.

    This transform expects C(Z)YX dimensions and supports:
    - Standard N2V masking on data channels
    - Asymmetric masking on auxiliary channels (lighter masking)
    - Channel dropout regularization

    Parameters
    ----------
    n2v_manipulate_config : N2VManipulateConfig
        N2V manipulation configuration.
    seed : Optional[int], optional
        Random seed, by default None.
    device : str
        The device on which operations take place, e.g. "cuda", "cpu" or "mps".

    Attributes
    ----------
    masked_pixel_percentage : float
        Percentage of pixels to mask in data channels.
    auxiliary_mask_percentage : float
        Percentage of pixels to mask in auxiliary channels.
    auxiliary_dropout_probability : float
        Probability of dropping an auxiliary channel per sample.
    roi_size : int
        Size of the replacement area.
    strategy : Literal[ "uniform", "median" ]
        Replacement strategy, uniform or median.
    remove_center : bool
        Whether to remove central pixel from patch.
    struct_mask : Optional[StructMaskParameters]
        StructN2V mask parameters.
    rng : Generator
        Random number generator.
    """

    def __init__(
        self,
        n2v_manipulate_config: N2VManipulateConfig,
        seed: int | None = None,
        device: str | None = None,
        n_data_channels: int = 1,
    ):
        """Constructor.

        Parameters
        ----------
        n2v_manipulate_config : N2VManipulateConfig
            N2V manipulation configuration.
        seed : Optional[int], optional
            Random seed, by default None.
        device : str
            The device on which operations take place, e.g. "cuda", "cpu" or "mps".
        n_data_channels : int
            Number of data channels (used when data_channel_indices is None in config).
        """
        # Determine which channels to mask
        if n2v_manipulate_config.data_channel_indices is not None:
            self.data_channel_indices = n2v_manipulate_config.data_channel_indices
        else:
            # Fallback to first n_data_channels
            n_channels = n2v_manipulate_config.n_data_channels
            self.data_channel_indices = list(range(n_channels))

        # Keep for backward compatibility
        self.n_data_channels = len(self.data_channel_indices)

        self.masked_pixel_percentage = n2v_manipulate_config.masked_pixel_percentage
        self.auxiliary_mask_percentage = n2v_manipulate_config.auxiliary_mask_percentage
        self.auxiliary_dropout_probability = n2v_manipulate_config.auxiliary_dropout_probability
        self.roi_size = n2v_manipulate_config.roi_size
        self.strategy = n2v_manipulate_config.strategy
        self.remove_center = n2v_manipulate_config.remove_center

        if n2v_manipulate_config.struct_mask_axis == SupportedStructAxis.NONE:
            self.struct_mask: StructMaskParameters | None = None
        else:
            self.struct_mask = StructMaskParameters(
                axis=(
                    0
                    if n2v_manipulate_config.struct_mask_axis
                    == SupportedStructAxis.HORIZONTAL
                    else 1
                ),
                span=n2v_manipulate_config.struct_mask_span,
            )

        # PyTorch random generator
        # TODO refactor into careamics.utils.torch_utils.get_device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available() and platform.processor() in (
                "arm",
                "arm64",
            ):
                device = "mps"
            else:
                device = "cpu"

        self.rng = (
            torch.Generator(device=device).manual_seed(seed)
            if seed is not None
            else torch.Generator(device=device)
        )

    def __call__(
        self, batch: torch.Tensor, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the transform to the image with asymmetric masking and channel dropout.

        Parameters
        ----------
        batch : torch.Tensor
            Batch if image patches, 2D or 3D, shape BC(Z)YX.
        *args : Any
            Additional arguments, unused.
        **kwargs : Any
            Additional keyword arguments, unused.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Masked patch, original patch, and mask.
        """
        batch_size = batch.shape[0]
        n_channels = batch.shape[1]

        masked = torch.zeros_like(batch)
        mask = torch.zeros_like(batch, dtype=torch.uint8)

        # Create a set of data channel indices for faster lookup
        data_channels_set = set(self.data_channel_indices)

        # Generate auxiliary channel dropout mask per sample
        # Shape: (batch_size, n_channels)
        if self.auxiliary_dropout_probability > 0:
            dropout_mask = torch.rand(
                batch_size, n_channels,
                generator=self.rng,
                device=batch.device
            ) < self.auxiliary_dropout_probability
        else:
            dropout_mask = torch.zeros(
                batch_size, n_channels,
                dtype=torch.bool,
                device=batch.device
            )

        if self.strategy == SupportedPixelManipulation.UNIFORM:
            # Iterate over batch and channels
            for b in range(batch_size):
                for c in range(n_channels):
                    # Check if this is a data channel
                    is_data_channel = c in data_channels_set

                    # Check if this auxiliary channel should be dropped
                    if not is_data_channel and dropout_mask[b, c]:
                        # Drop this auxiliary channel (set to zero)
                        masked[b, c, ...] = torch.zeros_like(batch[b, c, ...])
                        # No mask for dropped channels
                        continue

                    # Determine masking percentage
                    if is_data_channel:
                        mask_pct = self.masked_pixel_percentage
                    else:
                        mask_pct = self.auxiliary_mask_percentage

                    # Apply masking (skip if auxiliary_mask_percentage is 0)
                    if mask_pct > 0:
                        masked_result, mask_result = uniform_manipulate_torch(
                            patch=batch[b, c, ...],
                            mask_pixel_percentage=mask_pct,
                            subpatch_size=self.roi_size,
                            remove_center=self.remove_center,
                            struct_params=self.struct_mask,
                            rng=self.rng,
                        )
                        masked[b, c, ...] = masked_result

                        # Only set mask for data channels (used in loss computation)
                        if is_data_channel:
                            mask[b, c, ...] = mask_result
                        # For auxiliary channels: mask stays zero (no loss computed)
                    else:
                        # No masking, just copy
                        masked[b, c, ...] = batch[b, c, ...]

        elif self.strategy == SupportedPixelManipulation.MEDIAN:
            # Iterate over batch and channels
            for b in range(batch_size):
                for c in range(n_channels):
                    is_data_channel = c in data_channels_set

                    if not is_data_channel and dropout_mask[b, c]:
                        masked[b, c, ...] = torch.zeros_like(batch[b, c, ...])
                        continue

                    if is_data_channel:
                        mask_pct = self.masked_pixel_percentage
                    else:
                        mask_pct = self.auxiliary_mask_percentage

                    if mask_pct > 0:
                        masked_result, mask_result = median_manipulate_torch(
                            batch=batch[b, c, ...],
                            mask_pixel_percentage=mask_pct,
                            subpatch_size=self.roi_size,
                            struct_params=self.struct_mask,
                            rng=self.rng,
                        )
                        masked[b, c, ...] = masked_result

                        # Only set mask for data channels (used in loss computation)
                        if is_data_channel:
                            mask[b, c, ...] = mask_result
                        # For auxiliary channels: mask stays zero (no loss computed)
                    else:
                        masked[b, c, ...] = batch[b, c, ...]
        else:
            raise ValueError(f"Unknown masking strategy ({self.strategy}).")

        return masked, batch, mask
