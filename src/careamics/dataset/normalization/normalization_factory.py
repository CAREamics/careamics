"""Normalization factory."""

from careamics.config.data.normalization_config import NormalizationConfig
from careamics.config.support import SupportedNormalization

from .mean_std_normalization import MeanStdNormalization
from .no_normalization import NoNormalization
from .normalization import Normalization
from .range_normalization import RangeNormalization


def create_normalization(norm_model: NormalizationConfig) -> Normalization:
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
            return MeanStdNormalization(
                **norm_model.model_dump(exclude={"name", "per_channel"}),
            )
        case SupportedNormalization.QUANTILE:
            return RangeNormalization(
                input_mins=norm_model.input_lower_quantile_values,
                input_maxes=norm_model.input_upper_quantile_values,
                target_mins=norm_model.target_lower_quantile_values,
                target_maxes=norm_model.target_upper_quantile_values,
            )
        case SupportedNormalization.MINMAX:
            return RangeNormalization(
                **norm_model.model_dump(exclude={"name", "per_channel"}),
            )
        case SupportedNormalization.NONE:
            return NoNormalization()
        case _:
            raise ValueError(f"Unknown normalization strategy: {norm_model.name}")
