from typing import Any, Literal, Optional

import torch

from careamics.config.support import SupportedPixelManipulation, SupportedStructAxis

from .pixel_manipulation_torch import median_manipulate_torch, uniform_manipulate_torch
from .struct_mask_parameters import StructMaskParameters


class N2VManipulateTorch:
    """
    Default augmentation for the N2V model.

    This transform expects C(Z)YX dimensions.

    Parameters
    ----------
    roi_size : int, optional
        Size of the replacement area, by default 11.
    masked_pixel_percentage : float, optional
        Percentage of pixels to mask, by default 0.2.
    strategy : Literal[ "uniform", "median" ], optional
        Replacement strategy, uniform or median, by default uniform.
    remove_center : bool, optional
        Whether to remove central pixel from patch, by default True.
    struct_mask_axis : Literal["horizontal", "vertical", "none"], optional
        StructN2V mask axis, by default "none".
    struct_mask_span : int, optional
        StructN2V mask span, by default 5.
    seed : Optional[int], optional
        Random seed, by default None.

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
        roi_size: int = 11,
        masked_pixel_percentage: float = 0.2,
        strategy: Literal["uniform", "median"] = SupportedPixelManipulation.UNIFORM,
        remove_center: bool = True,
        struct_mask_axis: Literal["horizontal", "vertical", "none"] = "none",
        struct_mask_span: int = 5,
        seed: Optional[int] = None,
    ):
        """Constructor.

        Parameters
        ----------
        roi_size : int, optional
            Size of the replacement area, by default 11.
        masked_pixel_percentage : float, optional
            Percentage of pixels to mask, by default 0.2.
        strategy : Literal[ "uniform", "median" ], optional
            Replacement strategy, uniform or median, by default uniform.
        remove_center : bool, optional
            Whether to remove central pixel from patch, by default True.
        struct_mask_axis : Literal["horizontal", "vertical", "none"], optional
            StructN2V mask axis, by default "none".
        struct_mask_span : int, optional
            StructN2V mask span, by default 5.
        seed : Optional[int], optional
            Random seed, by default None.
        """
        self.masked_pixel_percentage = masked_pixel_percentage
        self.roi_size = roi_size
        self.strategy = strategy
        self.remove_center = remove_center

        if struct_mask_axis == SupportedStructAxis.NONE:
            self.struct_mask: Optional[StructMaskParameters] = None
        else:
            self.struct_mask = StructMaskParameters(
                axis=0 if struct_mask_axis == SupportedStructAxis.HORIZONTAL else 1,
                span=struct_mask_span,
            )

        # PyTorch random generator
        self.rng = (
            torch.Generator().manual_seed(seed)
            if seed is not None
            else torch.default_generator
        )

    def __call__(
        self, patch: torch.Tensor, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the transform to the image.

        Parameters
        ----------
        patch : torch.Tensor
            Image patch, 2D or 3D, shape C(Z)YX.
        *args : Any
            Additional arguments, unused.
        **kwargs : Any
            Additional keyword arguments, unused.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Masked patch, original patch, and mask.
        """
        masked = torch.zeros_like(patch)
        mask = torch.zeros_like(patch, dtype=torch.uint8)

        if self.strategy == SupportedPixelManipulation.UNIFORM:
            # Iterate over the channels to apply manipulation separately
            for c in range(patch.shape[0]):
                masked[c, ...], mask[c, ...] = uniform_manipulate_torch(
                    patch=patch[c, ...],
                    mask_pixel_percentage=self.masked_pixel_percentage,
                    subpatch_size=self.roi_size,
                    remove_center=self.remove_center,
                    struct_params=self.struct_mask,
                    rng=self.rng,
                )
        elif self.strategy == SupportedPixelManipulation.MEDIAN:
            # Iterate over the channels to apply manipulation separately
            for c in range(patch.shape[0]):
                masked[c, ...], mask[c, ...] = median_manipulate_torch(
                    patch=patch[c, ...],
                    mask_pixel_percentage=self.masked_pixel_percentage,
                    subpatch_size=self.roi_size,
                    struct_params=self.struct_mask,
                    rng=self.rng,
                )
        else:
            raise ValueError(f"Unknown masking strategy ({self.strategy}).")

        return masked, patch, mask
