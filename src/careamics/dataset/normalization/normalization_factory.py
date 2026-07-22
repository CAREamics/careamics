"""Normalization factory."""

from careamics.config.data.normalization_config import (
    MeanStdConfig,
    MinMaxConfig,
    NoNormConfig,
    NormalizationConfig,
    QuantileConfig,
)

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
    # from PEP 634, Class patterns
    # if no arguments are present, the pattern succeeds if
    # the isinstance() check succeeds.
    match norm_model:
        case MeanStdConfig():
            return MeanStdNormalization(
                **norm_model.model_dump(exclude={"name", "per_channel"}),
            )
        case QuantileConfig():
            if (
                norm_model.input_lower_quantile_values is None
                or norm_model.input_upper_quantile_values is None
            ):
                raise ValueError(
                    "Quantile values must be computed before creating the "
                    "normalization transform."
                )
            return RangeNormalization(
                input_mins=norm_model.input_lower_quantile_values,
                input_maxes=norm_model.input_upper_quantile_values,
                target_mins=norm_model.target_lower_quantile_values,
                target_maxes=norm_model.target_upper_quantile_values,
                skip_target=norm_model.skip_target,
            )
        case MinMaxConfig():
            return RangeNormalization(
                **norm_model.model_dump(exclude={"name", "per_channel"}),
            )
        case NoNormConfig():
            return NoNormalization()
        case _:
            raise ValueError(f"Unknown normalization strategy: {norm_model.name}")
