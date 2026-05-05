"""Configuration for PN2V."""

from careamics.config.algorithms import PN2VAlgorithm

from .n2v_configuration import N2VConfiguration


class PN2VConfiguration(N2VConfiguration):
    """PN2V-specific configuration with N2V-style mask validation."""

    algorithm_config: PN2VAlgorithm
