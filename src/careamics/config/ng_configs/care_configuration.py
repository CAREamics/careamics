"""Configuration for N2V."""

from careamics.config.algorithms import CAREAlgorithm

from .ng_configuration import NGConfiguration


class CAREConfiguration(NGConfiguration):
    """CARE-specific configuration."""

    algorithm_config: CAREAlgorithm
