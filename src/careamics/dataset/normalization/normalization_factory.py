"""Normalization factory."""

from careamics.config.data.normalization_config import NormalizationConfig
from careamics.config.support import SupportedNormalization

from .mean_std_normalization import MeanStdNormalization
from .no_normalization import NoNormalization
from .normalization_protocol import NormalizationProtocol
from .range_normalization import RangeNormalization


def create_normalization(norm_model: NormalizationConfig) -> NormalizationProtocol:
    """
    Build a normalization transform from a normalization model.

    Parameters
    ----------
    norm_model : NormalizationConfig
        The normalization configuration.

    Returns
    -------
    NormalizationProtocol
        The normalization transform.
    """
    match norm_model.name:
        case SupportedNormalization.MEAN_STD:
            return MeanStdNormalization(**norm_model.model_dump(exclude={"name"}))
        case SupportedNormalization.QUANTILE:
            return RangeNormalization(**norm_model.model_dump(exclude={"name"}))
        case SupportedNormalization.MINMAX:
            return RangeNormalization(**norm_model.model_dump(exclude={"name"}))
        case SupportedNormalization.NONE:
            return NoNormalization()
        case _:
            raise ValueError(f"Unknown normalization strategy: {norm_model.name}")
