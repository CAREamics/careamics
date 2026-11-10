"""Configuration for CryoCARE."""

from careamics.config.algorithms import CryoCAREAlgorithm
from careamics.config.data import CryoCAREDataConfig

from .configuration import Configuration


class CryoCAREConfiguration(Configuration):
    """CryoCARE-specific top-level configuration."""

    algorithm_config: CryoCAREAlgorithm
    """CryoCARE algorithm configuration."""

    data_config: CryoCAREDataConfig
    """CryoCARE data configuration."""
