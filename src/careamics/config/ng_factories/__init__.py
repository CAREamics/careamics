"""Convenience functions to create coherent configurations for CAREamics."""

__all__ = [
    "create_advanced_care_config",
    "create_advanced_n2n_config",
    "create_advanced_n2v_config",
    "create_advanced_seg_config",
    "create_care_config",
    "create_n2n_config",
    "create_n2v_config",
    "create_ng_data_configuration",
    "create_seg_config",
    "create_seg_configuration",
    "create_structn2v_config",
    "instantiate_algorithm_config",
    "instantiate_config",
]

from .care_n2n_factory import (
    create_advanced_care_config,
    create_advanced_n2n_config,
    create_care_config,
    create_n2n_config,
)
from .data_factory import create_ng_data_configuration
from .n2v_factory import (
    create_advanced_n2v_config,
    create_n2v_config,
    create_structn2v_config,
)
from .ng_config_discriminator import (
    instantiate_algorithm_config,
    instantiate_config,
)
from .seg_unet_factory import create_advanced_seg_config, create_seg_config
