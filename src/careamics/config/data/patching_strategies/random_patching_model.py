"""Random patching Pydantic model."""

from typing import Literal

from .patched_model import PatchedModel


class RandomPatchingModel(PatchedModel):
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


class FixedRandomPatchingModel(PatchedModel):
    """Fixed random patching Pydantic model.

    Attributes
    ----------
    name : "fixed_random"
        The name of the patching strategy.
    patch_size : sequence of int
        The size of the patch in each spatial dimension, each patch size must be a power
        of 2 and larger than 8.
    """

    name: Literal["fixed_random"] = "fixed_random"
    """The name of the patching strategy."""
