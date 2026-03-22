"""N2V manipulation transform for PyTorch."""

import platform
from typing import Any

import torch

from careamics.config.augmentations import N2VManipulateConfig
from careamics.config.support import SupportedPixelManipulation, SupportedStructAxis

from .pixel_manipulation_torch import (
    _apply_struct_mask_torch_vec,
    _create_center_pixel_mask,
    _create_struct_mask,
    _get_stratified_coords_torch,
    _get_subpatch_coords,
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
    device : str
        The device on which operations take place, e.g. "cuda", "cpu" or "mps".
    fast : bool
        If True, enable an optional fast path that caches invariant tensors and
        uses a vectorized struct-mask application (no Python loop over batch
        items). Functionally equivalent to the reference path but outputs are
        **not** bit-identical when ``struct_mask_axis != "none"`` because a
        different RNG call pattern is used for struct replacement values.
        Default is False (reference behavior preserved).

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
        device: str | None = None,
        fast: bool = False,
    ):
        """Constructor.

        Parameters
        ----------
        n2v_manipulate_config : N2VManipulateConfig
            N2V manipulation configuration.
        device : str
            The device on which operations take place, e.g. "cuda", "cpu" or "mps".
        fast : bool
            Enable the fast path with cached invariant tensors and vectorized
            struct-mask application. Default is False.
        """
        self.masked_pixel_percentage = n2v_manipulate_config.masked_pixel_percentage
        self.roi_size = n2v_manipulate_config.roi_size
        self.strategy = n2v_manipulate_config.strategy
        self.fast = fast

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

        self.rng = torch.Generator(device=device).manual_seed(
            n2v_manipulate_config.seed
        )

        # --- fast-path cached tensors ---
        # Pre-compute shape/config-independent tensors once at construction.
        # Shape-dependent entries (_subpatch_mask_cache) are populated lazily.
        if fast:
            roi_span_full = torch.arange(
                -(self.roi_size // 2),
                self.roi_size // 2 + 1,
                dtype=torch.int32,
                device=device,
            )
            # remove_center=True is the default and only supported path
            self._roi_span: torch.Tensor = roi_span_full[roi_span_full != 0]
            self._roi_span_min: int = int(self._roi_span.min())
            self._roi_span_max: int = int(self._roi_span.max())
            # keyed by ndims (spatial dimensionality) → subpatch boolean mask
            self._subpatch_mask_cache: dict[int, torch.Tensor] = {}

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
        if self.fast:
            return self._call_fast(batch)

        masked = torch.zeros_like(batch)
        mask = torch.zeros_like(batch, dtype=torch.uint8)

        if self.strategy == SupportedPixelManipulation.UNIFORM:
            # Iterate over the channels to apply manipulation separately
            for c in range(batch.shape[1]):
                masked[:, c, ...], mask[:, c, ...] = uniform_manipulate_torch(
                    patch=batch[:, c, ...],
                    mask_pixel_percentage=self.masked_pixel_percentage,
                    subpatch_size=self.roi_size,
                    struct_params=self.struct_mask,
                    rng=self.rng,
                )
        elif self.strategy == SupportedPixelManipulation.MEDIAN:
            # Iterate over the channels to apply manipulation separately
            for c in range(batch.shape[1]):
                masked[:, c, ...], mask[:, c, ...] = median_manipulate_torch(
                    batch=batch[:, c, ...],
                    mask_pixel_percentage=self.masked_pixel_percentage,
                    subpatch_size=self.roi_size,
                    struct_params=self.struct_mask,
                    rng=self.rng,
                )
        else:
            raise ValueError(f"Unknown masking strategy ({self.strategy}).")

        return masked, batch, mask

    def _call_fast(
        self, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fast path: cached invariant tensors + vectorized struct-mask application.

        Mirrors the logic of ``uniform_manipulate_torch`` / ``median_manipulate_torch``
        but uses pre-computed ``roi_span`` / ``subpatch_mask`` tensors and calls
        ``_apply_struct_mask_torch_vec`` instead of ``_apply_struct_mask_torch``.

        .. warning::
            If the reference implementations of ``uniform_manipulate_torch`` or
            ``median_manipulate_torch`` change, this method must be updated to match.

        Parameters
        ----------
        batch : torch.Tensor
            Batch of image patches, shape BC(Z)YX.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Masked patch, original patch, and mask.
        """
        masked = torch.zeros_like(batch)
        mask = torch.zeros_like(batch, dtype=torch.uint8)

        if self.strategy == SupportedPixelManipulation.UNIFORM:
            for c in range(batch.shape[1]):
                masked[:, c, ...], mask[:, c, ...] = self._uniform_fast(
                    batch[:, c, ...]
                )
        elif self.strategy == SupportedPixelManipulation.MEDIAN:
            for c in range(batch.shape[1]):
                masked[:, c, ...], mask[:, c, ...] = self._median_fast(
                    batch[:, c, ...]
                )
        else:
            raise ValueError(f"Unknown masking strategy ({self.strategy}).")

        return masked, batch, mask

    def _uniform_fast(
        self, patch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fast uniform manipulation using cached roi_span tensors.

        Mirrors ``uniform_manipulate_torch`` with ``remove_center=True``.
        """
        transformed_patch = patch.clone()

        subpatch_centers = _get_stratified_coords_torch(
            self.masked_pixel_percentage, patch.shape, self.rng
        ).to(device=patch.device)

        # Use cached roi_span (avoids torch.arange + filter each call)
        random_increment = self._roi_span[
            torch.randint(
                low=self._roi_span_min,
                high=self._roi_span_max + 1,
                size=(subpatch_centers.shape[0], subpatch_centers.shape[1] - 1),
                generator=self.rng,
                device=patch.device,
            )
        ]

        replacement_coords = subpatch_centers.clone()
        replacement_coords[:, 1:] = torch.clamp(
            replacement_coords[:, 1:] + random_increment,
            torch.zeros_like(torch.tensor(patch.shape[1:])).to(device=patch.device),
            torch.tensor([v - 1 for v in patch.shape[1:]]).to(device=patch.device),
        )

        replacement_pixels = patch[tuple(replacement_coords.T)]
        transformed_patch[tuple(subpatch_centers.T)] = replacement_pixels

        mask = (transformed_patch != patch).to(dtype=torch.uint8)

        if self.struct_mask is not None:
            transformed_patch = _apply_struct_mask_torch_vec(
                transformed_patch, subpatch_centers, self.struct_mask, self.rng
            )

        return transformed_patch, mask

    def _median_fast(
        self, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fast median manipulation using lazily cached subpatch_mask.

        Mirrors ``median_manipulate_torch``.
        """
        ndims = batch.ndim - 1

        subpatch_center_coordinates = _get_stratified_coords_torch(
            self.masked_pixel_percentage, batch.shape, self.rng
        )
        subpatch_coords = _get_subpatch_coords(
            subpatch_center_coordinates, self.roi_size, batch.shape
        )
        subpatches = batch[tuple(subpatch_coords)]

        # Lazily populate the subpatch_mask cache keyed by ndims
        if ndims not in self._subpatch_mask_cache:
            if self.struct_mask is None:
                self._subpatch_mask_cache[ndims] = _create_center_pixel_mask(
                    ndims, self.roi_size, batch.device
                )
            else:
                self._subpatch_mask_cache[ndims] = _create_struct_mask(
                    ndims, self.roi_size, self.struct_mask, batch.device
                )

        subpatch_mask = self._subpatch_mask_cache[ndims]
        subpatches_masked = subpatches[:, subpatch_mask]
        medians = subpatches_masked.median(dim=1).values

        output_batch = batch.clone()
        output_batch[tuple(subpatch_center_coordinates.T)] = medians
        mask = (batch != output_batch).to(torch.uint8)

        if self.struct_mask is not None:
            output_batch = _apply_struct_mask_torch_vec(
                output_batch, subpatch_center_coordinates, self.struct_mask, self.rng
            )

        return output_batch, mask
