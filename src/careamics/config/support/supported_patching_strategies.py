"""Patching strategies supported by Careamics."""

from careamics.utils import BaseEnum


class SupportedPatchingStrategy(str, BaseEnum):
    """Patching strategies supported by Careamics."""

    RANDOM = "random"
    """Random patching strategy."""

    SEQUENTIAL = "sequential"
    """Sequential patching strategy."""

    TILED = "tiled"
    """Tiled patching strategy, used during prediction."""

    WHOLE = "whole"
    """Whole image patching strategy, used during prediction."""
