"""CAREamics Pydantic configuration models.

To maintain clarity at the module level, we follow the following naming conventions:
`*_model` is specific for sub-configurations (e.g. architecture, data, algorithm),
while `*_configuration` is reserved for the main configuration models, including the
`Configuration` base class and its algorithm-specific child classes.
"""

__all__ = [
    "UNetBasedAlgorithm",
    "VAEBasedAlgorithm",
    "DataConfig",
    "Configuration",
    "CheckpointModel",
    "InferenceConfig",
    "load_configuration",
    "save_configuration",
    "TrainingConfig",
    "create_n2v_configuration",
    "create_n2n_configuration",
    "create_care_configuration",
    "GaussianMixtureNMConfig",
    "MultiChannelNMConfig",
    "LVAELossConfig",
    "N2VConfiguration",
    "N2NConfiguration",
    "CAREConfiguration",
    "AlgorithmFactory",
    "DataFactory",
    "ConfigurationFactory",
    "N2VDataConfig",
]
from .algorithms import UNetBasedAlgorithm, VAEBasedAlgorithm
from .callback_model import CheckpointModel
from .care_configuration import CAREConfiguration
from .configuration import Configuration
from .configuration_factory import (
    AlgorithmFactory,
    ConfigurationFactory,
    DataFactory,
    create_care_configuration,
    create_n2n_configuration,
    create_n2v_configuration,
)
from .configuration_io import load_configuration, save_configuration
from .data import DataConfig, N2VDataConfig
from .inference_model import InferenceConfig
from .loss_model import LVAELossConfig
from .n2n_configuration import N2NConfiguration
from .n2v_configuration import N2VConfiguration
from .nm_model import GaussianMixtureNMConfig, MultiChannelNMConfig
from .training_model import TrainingConfig
