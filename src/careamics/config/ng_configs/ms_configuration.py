"""Configuration for MicroSplit."""

from careamics.config.algorithms import MicroSplitAlgorithm
from careamics.lvae_training.dataset.config import MicroSplitDataConfig

from .ng_configuration import NGConfiguration


class MSConfiguration(NGConfiguration):
    """MicroSplit-specific configuration."""

    algorithm_config: MicroSplitAlgorithm

    data_config: MicroSplitDataConfig
