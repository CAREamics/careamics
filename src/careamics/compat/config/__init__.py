"""Deprecated configuration from CAREamics v0.1.0."""

from .configuration import Configuration
from .configuration_factories import (
    algorithm_factory,
    create_care_configuration,
    create_hdn_configuration,
    create_microsplit_configuration,
    create_n2n_configuration,
    create_n2v_configuration,
    create_pn2v_configuration,
)

__all__ = [
    "Configuration",
    "algorithm_factory",
    "create_care_configuration",
    "create_hdn_configuration",
    "create_microsplit_configuration",
    "create_n2n_configuration",
    "create_n2v_configuration",
    "create_pn2v_configuration",
]
