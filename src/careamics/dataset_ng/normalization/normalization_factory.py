from careamics.config.data.normalization_config import NormalizationConfig
from careamics.config.support import SupportedNormalization

from .no_normalization import NoNormalization
from .normalization_protocol import NormalizationProtocol
from .range_normalization import RangeNormalization
from .standardization import Standardize


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
        return Standardize(
            input_means=norm_model.input_means,
            input_stds=norm_model.input_stds,
            target_means=getattr(norm_model, "target_means", None),
            target_stds=getattr(norm_model, "target_stds", None),
        )
    elif norm_model.name == SupportedNormalization.QUANTILE:
        return RangeNormalization(
            input_mins=norm_model.input_lower_quantile_values,
            input_maxes=norm_model.input_upper_quantile_values,
            target_mins=getattr(norm_model, "target_lower_quantile_values", None),
            target_maxes=getattr(norm_model, "target_upper_quantile_values", None),
        )
    elif norm_model.name == SupportedNormalization.MINMAX:
        return RangeNormalization(
            input_mins=norm_model.input_mins,
            input_maxes=norm_model.input_maxes,
            target_mins=getattr(norm_model, "target_mins", None),
            target_maxes=getattr(norm_model, "target_maxes", None),
        )
    elif norm_model.name == SupportedNormalization.NONE:
        return NoNormalization()
    else:
        raise ValueError(f"Unknown normalization strategy: {norm_model.name}")
