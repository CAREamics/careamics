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
    Default augmentation for the N2V model.

    This transform expects C(Z)YX dimensions.

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
        Percentage of pixels to mask.
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
        """Apply the transform to the image.

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
        masked = torch.zeros_like(batch)
        mask = torch.zeros_like(batch, dtype=torch.uint8)

        # Create a set of data channel indices for faster lookup
        data_channels_set = set(self.data_channel_indices)

        if self.strategy == SupportedPixelManipulation.UNIFORM:
            # Iterate over all channels
            for c in range(batch.shape[1]):
                if c in data_channels_set:
                    # Mask data channels
                    masked[:, c, ...], mask[:, c, ...] = uniform_manipulate_torch(
                        patch=batch[:, c, ...],
                        mask_pixel_percentage=self.masked_pixel_percentage,
                        subpatch_size=self.roi_size,
                        remove_center=self.remove_center,
                        struct_params=self.struct_mask,
                        rng=self.rng,
                    )
                else:
                    # Copy auxiliary channels without masking
                    masked[:, c, ...] = batch[:, c, ...]

        elif self.strategy == SupportedPixelManipulation.MEDIAN:
            # Iterate over all channels
            for c in range(batch.shape[1]):
                if c in data_channels_set:
                    # Mask data channels
                    masked[:, c, ...], mask[:, c, ...] = median_manipulate_torch(
                        batch=batch[:, c, ...],
                        mask_pixel_percentage=self.masked_pixel_percentage,
                        subpatch_size=self.roi_size,
                        struct_params=self.struct_mask,
                        rng=self.rng,
                    )
                else:
                    # Copy auxiliary channels without masking
                    masked[:, c, ...] = batch[:, c, ...]
        else:
            raise ValueError(f"Unknown masking strategy ({self.strategy}).")

        return masked, batch, mask
