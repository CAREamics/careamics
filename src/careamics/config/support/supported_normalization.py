"""Supported normalization strategies for CAREamics."""

from careamics.utils import BaseEnum


class SupportedNormalization(str, BaseEnum):
    """Normalization strategies supported by Careamics."""

    MEAN_STD = "mean_std"
    """Mean and std normalization strategy."""

    QUANTILE = "quantile"
    """Quantile normalization strategy."""

    MINMAX = "minmax"
    """Min-max normalization strategy."""

    NONE = "none"
    """No normalization strategy."""
