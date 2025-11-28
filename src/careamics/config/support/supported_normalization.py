from careamics.utils import BaseEnum


class SupportedNormalization(str, BaseEnum):
    """Normalization strategies supported by Careamics."""

    MEAN_STD = "standardize"
    """Mean and std normalization strategy."""

    QUANTILE = "quantile"
    """Quantile normalization strategy."""

    MINMAX = "minmax"
    """Min-max normalization strategy."""

    NONE = "none"
    """No normalization strategy."""
