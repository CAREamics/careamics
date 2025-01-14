"""CAREamics Pydantic configuration models.

To maintain clarity at the module level, we follow the following naming conventions:
`*_model` is specific for sub-configurations (e.g. architecture, data, algorithm),
while `*_configuration` is reserved for the main configuration models, including the
`Configuration` base class and its algorithm-specific child classes.
"""

__all__ = [
    "CAREAlgorithm",
    "CAREConfiguration",
    "CheckpointModel",
    "Configuration",
    "DataConfig",
    "GaussianMixtureNMConfig",
    "GeneralDataConfig",
    "InferenceConfig",
    "LVAELossConfig",
    "MultiChannelNMConfig",
    "N2NAlgorithm",
    "N2NConfiguration",
    "N2VAlgorithm",
    "N2VConfiguration",
    "N2VDataConfig",
    "TrainingConfig",
    "UNetBasedAlgorithm",
    "VAEBasedAlgorithm",
    "algorithm_factory",
    "configuration_factory",
    "create_care_configuration",
    "create_n2n_configuration",
    "create_n2v_configuration",
    "data_factory",
    "load_configuration",
    "save_configuration",
]

from .algorithms import (
    CAREAlgorithm,
    N2NAlgorithm,
    N2VAlgorithm,
    UNetBasedAlgorithm,
    VAEBasedAlgorithm,
)
from .callback_model import CheckpointModel
from .care_configuration import CAREConfiguration
from .configuration import Configuration
from .configuration_factories import (
    algorithm_factory,
    configuration_factory,
    create_care_configuration,
    create_n2n_configuration,
    create_n2v_configuration,
    data_factory,
)
from .configuration_io import load_configuration, save_configuration
from .data import DataConfig, GeneralDataConfig, N2VDataConfig
from .inference_model import InferenceConfig
from .loss_model import LVAELossConfig
from .n2n_configuration import N2NConfiguration
from .n2v_configuration import N2VConfiguration
from .nm_model import GaussianMixtureNMConfig, MultiChannelNMConfig
from .training_model import TrainingConfig
