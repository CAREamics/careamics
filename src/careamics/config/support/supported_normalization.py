"""Supported normalization strategies for CAREamics."""

from enum import StrEnum


class SupportedNormalization(StrEnum):
    """Normalization strategies supported by Careamics."""

    MEAN_STD = "mean_std"
    """Mean and std normalization strategy."""

    QUANTILE = "quantile"
    """Quantile normalization strategy."""

    MINMAX = "min_max"
    """Min-max normalization strategy."""

    NONE = "none"
    """No normalization strategy."""
