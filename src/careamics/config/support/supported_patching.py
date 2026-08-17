"""Patching strategies supported by Careamics."""

from enum import StrEnum


class SupportedPatching(StrEnum):
    """Patching strategies supported by Careamics."""

    FIXED_RANDOM = "fixed_random"
    """Fixed random patching strategy, used during training."""

    RANDOM = "random"
    """Random patching strategy, used during training."""

    STRATIFIED = "stratified"

    # SEQUENTIAL = "sequential"
    # """Sequential patching strategy, used during training."""

    TILED = "tiled"
    """Tiled patching strategy, used during prediction."""

    SLIDING_WINDOW_TILED = "sliding_window_tiled"
    """Sliding-window inner-tiled patching with stride decoupled from overlap.
    Used during prediction with posterior models."""

    WHOLE = "whole"
    """Whole image patching strategy, used during prediction."""
