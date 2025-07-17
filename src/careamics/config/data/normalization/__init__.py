"""Normalization strategies."""

from typing import Union

from .normalization_strategies import MeanStdNormModel, NoNormModel, QuantileNormModel

NormalizationStrategies = Union[
    MeanStdNormModel,
    NoNormModel,
    QuantileNormModel,
]

__all__ = [
    "MeanStdNormModel",
    "NoNormModel",
    "NormalizationStrategies",
    "QuantileNormModel",
]
