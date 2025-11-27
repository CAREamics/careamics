from .no_normalization import NoNormalization
from .normalization_protocol import NormalizationProtocol
from .standardization import Standardize
from .range_normalization import RangeNormalization

from careamics.config.transformations.transform_unions import NORM_MODES
from careamics.config.support import SupportedNormalization


def create_normalization(norm_model: NORM_MODES) -> NormalizationProtocol:
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
            image_means=norm_model.image_means,
            image_stds=norm_model.image_stds,
            target_means=getattr(norm_model, "target_means", None),
            target_stds=getattr(norm_model, "target_stds", None),
        )
    elif norm_model.name == SupportedNormalization.QUANTILE:
        return RangeNormalization(
            image_mins=norm_model.lower_quantiles,
            image_maxs=norm_model.upper_quantiles,
        )
    elif norm_model.name == SupportedNormalization.MINMAX:
        return RangeNormalization(
            image_mins=norm_model.image_mins,
            image_maxs=norm_model.image_maxs,
        )
    elif norm_model.name == SupportedNormalization.NONE:
        return NoNormalization()
    else:
        raise ValueError(f"Unknown normalization strategy: {norm_model.name}")
