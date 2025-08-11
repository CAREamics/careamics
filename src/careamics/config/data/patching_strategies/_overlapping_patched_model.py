"""Sequential patching Pydantic model."""

from collections.abc import Sequence

from pydantic import Field, ValidationInfo, field_validator

from ._patched_model import _PatchedModel


class _OverlappingPatchedModel(_PatchedModel):
    """Overlapping patching Pydantic model.

    This model is only used for inheritance and validation purposes.

    Attributes
    ----------
    patch_size : list of int
        The size of the patch in each spatial dimension, each patch size must be a power
        of 2 and larger than 8.
    overlaps : sequence of int, optional
        The overlaps between patches in each spatial dimension. If `None`, no overlap is
        applied. The overlaps must be smaller than the patch size in each spatial
        dimension, and the number of dimensions be either 2 or 3.
    """

    overlaps: Sequence[int] | None = Field(
        default=None,
        min_length=2,
        max_length=3,
    )
    """The overlaps between patches in each spatial dimension. If `None`, no overlap is
    applied. The overlaps must be smaller than the patch size in each spatial dimension,
    and the number of dimensions be either 2 or 3.
    """

    @field_validator("overlaps")
    @classmethod
    def overlap_smaller_than_patch_size(
        cls, overlaps: Sequence[int] | None, values: ValidationInfo
    ) -> Sequence[int] | None:
        """
        Validate overlap.

        Overlaps must be smaller than the patch size in each spatial dimension.

        Parameters
        ----------
        overlaps : Sequence of int
            Overlap in each dimension.
        values : ValidationInfo
            Dictionary of values.

        Returns
        -------
        Sequence of int
            Validated overlap.
        """
        if overlaps is None:
            return None

        patch_size = values.data["patch_size"]

        if len(overlaps) != len(patch_size):
            raise ValueError(
                f"Overlaps must have the same number of dimensions as the patch size. "
                f"Got {len(overlaps)} dimensions for overlaps and {len(patch_size)} "
                f"dimensions for patch size."
            )

        if any(o >= p for o, p in zip(overlaps, patch_size, strict=False)):
            raise ValueError(
                f"Overlap must be smaller than the patch size, got {overlaps} versus "
                f"{patch_size}."
            )

        return overlaps

    @field_validator("overlaps")
    @classmethod
    def overlap_even(cls, overlaps: Sequence[int] | None) -> Sequence[int] | None:
        """
        Validate overlaps.

        Overlap must be even.

        Parameters
        ----------
        overlaps : Sequence of int
            Overlaps.

        Returns
        -------
        Sequence of int
            Validated overlap.
        """
        if overlaps is None:
            return None

        if any(o % 2 != 0 for o in overlaps):
            raise ValueError(f"Overlaps must be even, got {overlaps}.")

        return overlaps
