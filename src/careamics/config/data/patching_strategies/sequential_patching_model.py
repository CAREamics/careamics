"""Sequential patching Pydantic model."""

from typing import Literal

from .overlapping_patched_model import OverlappingPatchedModel


class SequentialPatchingModel(OverlappingPatchedModel):
    """Sequential patching Pydantic model."""

    name: Literal["sequential"] = "sequential"
    """The name of the patching strategy."""
