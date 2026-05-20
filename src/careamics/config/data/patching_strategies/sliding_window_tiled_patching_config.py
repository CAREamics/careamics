"""Sliding-window tiled patching Pydantic model."""

from collections.abc import Sequence
from typing import Literal

from pydantic import Field, ValidationInfo, field_validator

from ._overlapping_patched_config import _OverlappingPatchedConfig


class SlidingWindowTiledPatchingConfig(_OverlappingPatchedConfig):
    """Sliding-window inner-tiled patching Pydantic model.

    Decouples `stride` from `overlaps` to enable dense overlap of inner-tiled
    kept regions. Used at prediction time by `SlidingWindowTiledPatching`.

    Attributes
    ----------
    name : "sliding_window_tiled"
        The name of the patching strategy.
    patch_size : sequence of int
        Tile size per spatial dimension.
    overlaps : sequence of int
        Overlap per spatial dimension (= 2 * margin). Must be even and smaller
        than the patch size.
    stride : sequence of int
        Tile stride per spatial dimension. Must be positive and satisfy
        `stride[i] <= patch_size[i] - overlaps[i]`.
    """

    name: Literal["sliding_window_tiled"] = "sliding_window_tiled"
    """The name of the patching strategy."""

    stride: Sequence[int] = Field(
        ...,
        min_length=2,
        max_length=3,
    )
    """Tile stride per spatial dimension."""

    @field_validator("stride")
    @classmethod
    def stride_compatible_with_patch_overlap(
        cls, stride: Sequence[int], info: ValidationInfo
    ) -> Sequence[int]:
        """Validate `stride` against `patch_size` and `overlaps`.

        Each axis must satisfy `0 < stride[i] <= patch_size[i] - overlaps[i]`.
        If `overlaps` is None, treats it as all-zeros (degenerate sliding
        window without inner cropping).
        """
        patch_size = info.data.get("patch_size")
        overlaps = info.data.get("overlaps")
        if patch_size is None:
            raise ValueError(
                "Cannot validate stride because of undefined patch_size. There "
                "may be validation errors in `patch_size`."
            )
        if len(stride) != len(patch_size):
            raise ValueError(
                f"Stride must have the same number of dimensions as patch_size. "
                f"Got {len(stride)} dimensions for stride and {len(patch_size)} "
                f"for patch_size."
            )
        if any(s <= 0 for s in stride):
            raise ValueError(f"Stride must be strictly positive, got {stride}.")

        effective_overlaps = (
            overlaps if overlaps is not None else [0] * len(patch_size)
        )
        for i, (p, o, s) in enumerate(
            zip(patch_size, effective_overlaps, stride, strict=True)
        ):
            if s > p - o:
                raise ValueError(
                    f"Axis {i}: stride ({s}) must be <= patch_size - overlap "
                    f"({p} - {o} = {p - o})."
                )
        return stride
