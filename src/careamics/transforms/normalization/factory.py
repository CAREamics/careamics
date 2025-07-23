"""Normalization transform factory."""

from .no_normalization import NoNormalization
from .normalization_protocol import NormalizationProtocol
from .standardization import Standardize


def build_normalization_transform(norm_model) -> NormalizationProtocol:
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
    if norm_model.name == "standard":
        return Standardize(
            image_means=norm_model.image_means,
            image_stds=norm_model.image_stds,
            target_means=getattr(norm_model, "target_means", None),
            target_stds=getattr(norm_model, "target_stds", None),
        )
    elif norm_model.name == "none":
        return NoNormalization()
    else:
        raise ValueError(f"Unknown normalization strategy: {norm_model.name}")
