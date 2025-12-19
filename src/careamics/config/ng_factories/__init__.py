"""Convenience functions to create coherent configurations for CAREamics."""

__all__ = [
    "create_n2v_configuration",
    "create_ng_data_configuration",
]

from .data_factory import create_ng_data_configuration
from .n2v_factory import create_n2v_configuration
