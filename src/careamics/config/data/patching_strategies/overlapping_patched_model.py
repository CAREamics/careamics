"""Sequential patching Pydantic model."""

from collections.abc import Sequence
from typing import Optional

from pydantic import Field, ValidationInfo, field_validator

from .patched_model import PatchedModel


class OverlappingPatchedModel(PatchedModel):
    """Overlapping patching Pydantic model."""

    overlap: Optional[Sequence[int]] = Field(
        default=None,
        min_length=2,
        max_length=3,
    )
    """The overlap between patches in each spatial dimension, if None, no overlap is
    applied. The overlap must be smaller than the patch size in each spatial dimension.
    """

    @field_validator("overlap")
    @classmethod
    def overlap_smaller_than_patch_size(
        cls, overlap: Optional[Sequence[int]], values: ValidationInfo
    ) -> Optional[Sequence[int]]:
        """
        Validate overlap.

        Overlap must be smaller than the patch size in each spatial dimension.

        Parameters
        ----------
        overlap : Sequence of int
            Overlap.
        values : ValidationInfo
            Dictionary of values.

        Returns
        -------
        Sequence of int
            Validated overlap.
        """
        if overlap is None:
            return None

        patch_size = values.data["patch_size"]

        if len(overlap) != len(patch_size):
            raise ValueError(
                f"Overlap must have the same number of dimensions as the patch size. "
                f"Got {len(overlap)} dimensions for overlap and {len(patch_size)} "
                f"dimensions for patch size."
            )

        if any(o >= p for o, p in zip(overlap, patch_size)):
            raise ValueError("Overlap must be smaller than the patch size.")

        return overlap

    @field_validator("overlap")
    @classmethod
    def overlap_even(cls, overlap: Optional[Sequence[int]]) -> Optional[Sequence[int]]:
        """
        Validate overlap.

        Overlap must be even.

        Parameters
        ----------
        overlap : Sequence of int
            Overlap.

        Returns
        -------
        Sequence of int
            Validated overlap.
        """
        if overlap is None:
            return None

        if any(o % 2 != 0 for o in overlap):
            raise ValueError("Overlap must be even.")

        return overlap
