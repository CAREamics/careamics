"""Random patching Pydantic model."""

from typing import Literal

from .patched_model import PatchedModel


class RandomPatchingModel(PatchedModel):
    """Random patching Pydantic model."""

    name: Literal["random"] = "random"
    """The name of the patching strategy."""
