"""Deprecated."""

__all__ = [
    "NORM_AND_SPATIAL_UNION",
    "SPATIAL_TRANSFORMS_UNION",
    "NormalizeConfig",
    "TransformConfig",
]


from .normalize_config import NormalizeConfig
from .transform_config import TransformConfig
from .transform_unions import (
    NORM_AND_SPATIAL_UNION,
    SPATIAL_TRANSFORMS_UNION,
)
