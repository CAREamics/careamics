"""Configuration for N2N."""

from careamics.config.algorithms import N2NAlgorithm

from .ng_configuration import NGConfiguration


class N2NConfiguration(NGConfiguration):
    """N2N-specific configuration."""

    algorithm_config: N2NAlgorithm
