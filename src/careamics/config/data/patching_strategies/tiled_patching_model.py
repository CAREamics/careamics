"""Tiled patching Pydantic model."""

from collections.abc import Sequence
from typing import Literal

from pydantic import Field

from ._overlapping_patched_model import _OverlappingPatchedModel


# TODO with UNet tiling must obey different rules than sequential tiling
#   - needs to validated at the level of the configuration
class TiledPatchingModel(_OverlappingPatchedModel):
    """Tiled patching Pydantic model.

    Attributes
    ----------
    name : "tiled"
        The name of the patching strategy.
    patch_size : sequence of int
        The size of the patch in each spatial dimension, each patch size must be a power
        of 2 and larger than 8.
    overlaps : sequence of int
        The overlaps between patches in each spatial dimension. The overlaps must be
        smaller than the patch size in each spatial dimension, and the number of
        dimensions be either 2 or 3.
    """

    name: Literal["tiled"] = "tiled"
    """The name of the patching strategy."""

    overlaps: Sequence[int] = Field(
        ...,
        min_length=2,
        max_length=3,
    )
    """The overlaps between patches in each spatial dimension. The overlaps must be
    smaller than the patch size in each spatial dimension, and the number of dimensions
    be either 2 or 3.
    """
