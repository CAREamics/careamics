"""Configuration module."""


__all__ = [
    "AlgorithmModel", "Configuration", "load_configuration", "save_configuration"
]

from .config import (
    Configuration,
    load_configuration,
    save_configuration,
)
from .algorithm import AlgorithmModel
