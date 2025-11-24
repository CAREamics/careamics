"""Random patching Pydantic model."""

from typing import Literal

from ._patched_config import _PatchedConfig


class RandomPatchingConfig(_PatchedConfig):
    """Random patching Pydantic model.

    Attributes
    ----------
    name : "random"
        The name of the patching strategy.
    patch_size : sequence of int
        The size of the patch in each spatial dimension, each patch size must be a power
        of 2 and larger than 8.
    """

    name: Literal["random"] = "random"
    """The name of the patching strategy."""
