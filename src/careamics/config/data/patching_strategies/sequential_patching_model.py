"""Sequential patching Pydantic model."""

from typing import Literal

from ._overlapping_patched_model import _OverlappingPatchedModel


class SequentialPatchingModel(_OverlappingPatchedModel):
    """Sequential patching Pydantic model.

    Attributes
    ----------
    name : "sequential"
        The name of the patching strategy.
    patch_size : sequence of int
        The size of the patch in each spatial dimension, each patch size must be a power
        of 2 and larger than 8.
    overlaps : list of int, optional
        The overlaps between patches in each spatial dimension. If `None`, no overlap is
        applied. The overlaps must be smaller than the patch size in each spatial
        dimension, and the number of dimensions be either 2 or 3.
    """

    name: Literal["sequential"] = "sequential"
    """The name of the patching strategy."""
