"""Tiled patching Pydantic model."""

from collections.abc import Sequence
from typing import Literal

from pydantic import Field, ValidationInfo, field_validator

from .patched_model import PatchedModel


class TiledPatchingModel(PatchedModel):
    """Tiled patching Pydantic model."""

    name: Literal["tiled"] = "tiled"
    """The name of the patching strategy."""

    overlap: Sequence[int] = Field(
        default=...,
        min_length=2,
        max_length=3,
    )
    """The overlap between patches in each spatial dimension. The overlap must be
    smaller than the patch size in each spatial dimension."""

    @field_validator("overlap")
    @classmethod
    def overlap_smaller_than_patch_size(
        cls, overlap: Sequence[int], values: ValidationInfo
    ) -> Sequence[int]:
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
        patch_size: Sequence[int] = values.data["patch_size"]
        if any(o >= p for o, p in zip(overlap, patch_size)):
            raise ValueError("Overlap must be smaller than the patch size.")

        return overlap

    @field_validator("overlap")
    @classmethod
    def overlap_even(cls, overlap: Sequence[int]) -> Sequence[int]:
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
        if any(o % 2 != 0 for o in overlap):
            raise ValueError("Overlap must be even.")

        return overlap
