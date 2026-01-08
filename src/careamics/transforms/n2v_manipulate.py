"""N2V manipulation transform."""

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from careamics.config.support import SupportedPixelManipulation, SupportedStructAxis
from careamics.transforms.transform import Transform

from .pixel_manipulation import median_manipulate, uniform_manipulate
from .struct_mask_parameters import StructMaskParameters


class N2VManipulate(Transform):
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
    data_channel_indices : Optional[list[int]], optional
        List of channel indices to apply manipulation to, by default None (all).
    """

    def __init__(
        self,
        roi_size: int = 11,
        masked_pixel_percentage: float = 0.2,
        strategy: Literal[
            "uniform", "median"
        ] = SupportedPixelManipulation.UNIFORM.value,
        remove_center: bool = True,
        struct_mask_axis: Literal["horizontal", "vertical", "none"] = "none",
        struct_mask_span: int = 5,
        seed: int | None = None,
        data_channel_indices: list[int] | None = None,  # <--- Added this argument
        n_data_channels: int | None = None,
    ):
        """Constructor."""
        super().__init__()

        self.masked_pixel_percentage = masked_pixel_percentage
        self.roi_size = roi_size
        self.strategy = strategy
        self.remove_center = remove_center
        self.data_channel_indices = data_channel_indices  # <--- Store it
        self.data_channel_indices = data_channel_indices
        if struct_mask_axis == SupportedStructAxis.NONE:
            self.struct_mask: StructMaskParameters | None = None
        else:
            self.struct_mask = StructMaskParameters(
                axis=0 if struct_mask_axis == SupportedStructAxis.HORIZONTAL else 1,
                span=struct_mask_span,
            )

        # numpy random generator
        self.rng = np.random.default_rng(seed=seed)

    def __call__(
        self, patch: NDArray, *args: Any, **kwargs: Any
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Apply the transform to the image.

        Parameters
        ----------
        patch : np.ndarray
            Image patch, 2D or 3D, shape C(Z)YX.
        """
        masked = np.zeros_like(patch)
        mask = np.zeros_like(patch)

        # Determine which channels to iterate over
        # If indices are provided, only loop over those. Otherwise, do all.
        n_channels = patch.shape[0]
        channels_to_process = (
            self.data_channel_indices
            if self.data_channel_indices is not None
            else range(n_channels)
        )

        # First copy original data to masked (so skipped channels remain untouched)
        masked[...] = patch[...]

        if self.strategy == SupportedPixelManipulation.UNIFORM:
            for c in channels_to_process:
                # Safety check in case config has bad indices
                if c >= n_channels:
                    continue

                masked[c, ...], mask[c, ...] = uniform_manipulate(
                    patch=patch[c, ...],
                    mask_pixel_percentage=self.masked_pixel_percentage,
                    subpatch_size=self.roi_size,
                    remove_center=self.remove_center,
                    struct_params=self.struct_mask,
                    rng=self.rng,
                )
        elif self.strategy == SupportedPixelManipulation.MEDIAN:
            for c in channels_to_process:
                if c >= n_channels:
                    continue

                masked[c, ...], mask[c, ...] = median_manipulate(
                    patch=patch[c, ...],
                    mask_pixel_percentage=self.masked_pixel_percentage,
                    subpatch_size=self.roi_size,
                    struct_params=self.struct_mask,
                    rng=self.rng,
                )
        else:
            raise ValueError(f"Unknown masking strategy ({self.strategy}).")

        return masked, patch, mask
