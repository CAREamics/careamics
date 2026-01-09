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
    if norm_model.name == SupportedNormalization.MEAN_STD:
        return MeanStdNormalization(
            input_means=norm_model.input_means,
            input_stds=norm_model.input_stds,
            target_means=norm_model.target_means,
            target_stds=norm_model.target_stds,
        )
    elif norm_model.name == SupportedNormalization.QUANTILE:
        return RangeNormalization(
            input_mins=norm_model.input_lower_quantile_values,
            input_maxes=norm_model.input_upper_quantile_values,
            target_mins=norm_model.target_lower_quantile_values,
            target_maxes=norm_model.target_upper_quantile_values,
        )
    elif norm_model.name == SupportedNormalization.MINMAX:
        return RangeNormalization(
            input_mins=norm_model.input_mins,
            input_maxes=norm_model.input_maxes,
            target_mins=norm_model.target_mins,
            target_maxes=norm_model.target_maxes,
        )
    elif norm_model.name == SupportedNormalization.NONE:
        return NoNormalization()
    else:
        raise ValueError(f"Unknown normalization strategy: {norm_model.name}")
