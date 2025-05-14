"""Tiled patching Pydantic model."""

from typing import Literal

from .overlapping_patched_model import OverlappingPatchedModel


# TODO with UNet tiling must obey different rules than sequential tiling
class TiledPatchingModel(OverlappingPatchedModel):
    """Tiled patching Pydantic model."""

    name: Literal["tiled"] = "tiled"
    """The name of the patching strategy."""
