from careamics.config.data.normalization_config import (
    MinMaxModel,
    NoNormModel,
    QuantileModel,
    StandardizeModel,
)
from careamics.config.support import SupportedNormalization

from .no_normalization import NoNormalization
from .normalization_protocol import NormalizationProtocol
from .range_normalization import RangeNormalization
from .standardization import Standardize

NormalizationModels = StandardizeModel | NoNormModel | QuantileModel | MinMaxModel


def create_normalization(norm_model: NormalizationModels) -> NormalizationProtocol:
    """
    Build a normalization transform from a normalization model.

    Parameters
    ----------
    norm_model : dict
        The normalization model.

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
            input_mins=norm_model.input_lower_quantiles,
            input_maxs=norm_model.input_upper_quantiles,
            target_mins=getattr(norm_model, "target_lower_quantiles", None),
            target_maxs=getattr(norm_model, "target_upper_quantiles", None),
        )
    elif norm_model.name == SupportedNormalization.MINMAX:
        return RangeNormalization(
            input_mins=norm_model.input_mins,
            input_maxs=norm_model.input_maxs,
            target_mins=getattr(norm_model, "target_mins", None),
            target_maxs=getattr(norm_model, "target_maxs", None),
        )
    elif norm_model.name == SupportedNormalization.NONE:
        return NoNormalization()
    else:
        raise ValueError(f"Unknown normalization strategy: {norm_model.name}")
