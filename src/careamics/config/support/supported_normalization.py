"""Normalization strategies supported by Careamics."""

from careamics.utils import BaseEnum


class SupportedNormalizationStrategy(str, BaseEnum):
    """Normalization strategies supported by Careamics."""

    MEAN_STD = "mean_std"
    """Mean and std normalization strategy."""

    QUANTILE = "quantile"
    """Quantile normalization strategy."""

    NONE = "none"
    """No normalization strategy."""
