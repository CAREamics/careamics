"""Patching strategies supported by Careamics."""

from careamics.utils import BaseEnum


class SupportedPatchingStrategy(str, BaseEnum):
    """Patching strategies supported by Careamics."""

    FIXED_RANDOM = "fixed_random"
    """Fixed random patching strategy, used during training."""

    RANDOM = "random"
    """Random patching strategy, used during training."""

    # SEQUENTIAL = "sequential"
    # """Sequential patching strategy, used during training."""

    TILED = "tiled"
    """Tiled patching strategy, used during prediction."""

    WHOLE = "whole"
    """Whole image patching strategy, used during prediction."""
