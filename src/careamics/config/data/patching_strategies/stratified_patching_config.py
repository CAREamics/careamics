"""Stratified patching Pydantic model."""

from typing import Literal

from pydantic import Field

from ._patched_config import _PatchedConfig


class StratifiedPatchingConfig(_PatchedConfig):
    """Stratified patching Pydantic model.

    Attributes
    ----------
    name : "stratified"
        The name of the patching strategy.
    patch_size : sequence of int
        The size of the patch in each spatial dimension, each patch size must be a power
        of 2 and larger than 8.
    """

    name: Literal["stratified"] = "stratified"
    """The name of the patching strategy."""

    seed: int | None = Field(default=None, gt=0)
    """Random seed for patch sampling, set to None for random seeding."""
