"""Discriminator for NG configurations."""

from typing import Annotated, Any, Union

from pydantic import Discriminator, Tag, TypeAdapter

from careamics.config.algorithms import (
    CAREAlgorithm,
    N2NAlgorithm,
    N2VAlgorithm,
    SegAlgorithm,
)
from careamics.config.data.normalization_config import NormalizationConfig
from careamics.config.ng_configs import N2VConfiguration
from careamics.config.ng_configs.ng_configuration import NGConfiguration
from careamics.config.support import SupportedAlgorithm


def _config_disciminator(v: Any) -> SupportedAlgorithm | None:
    """
    Extract algorithm type from configuration dict for Pydantic discriminator.

    Parameters
    ----------
    v : Any
        Configuration dictionary.

    Returns
    -------
    SupportedAlgorithm or None
        Algorithm type if found, None otherwise.
    """
    if not isinstance(v, dict):
        return None
    alg_config = v.get("algorithm_config", None)
    if not isinstance(alg_config, dict):
        return None
    return alg_config.get("algorithm", None)


def _algo_disciminator(algo: Any) -> SupportedAlgorithm | None:
    """
    Extract algorithm type from dict for Pydantic discriminator.

    Parameters
    ----------
    algo : Any
        Configuration dictionary.

    Returns
    -------
    SupportedAlgorithm or None
        Algorithm type if found, None otherwise.
    """
    if not isinstance(algo, dict):
        return None
    return algo.get("algorithm", None)


# ------------------------ Unions --------------------------

NGConfigs = Annotated[
    Union[
        Annotated[N2VConfiguration, Tag(SupportedAlgorithm.N2V)],
        Annotated[NGConfiguration, Tag(SupportedAlgorithm.CARE)],
        Annotated[NGConfiguration, Tag(SupportedAlgorithm.N2N)],
        Annotated[NGConfiguration, Tag(SupportedAlgorithm.SEG)],
    ],
    Discriminator(_config_disciminator),
]

NGAlgos = Annotated[
    Union[
        Annotated[N2VAlgorithm, Tag(SupportedAlgorithm.N2V)],
        Annotated[CAREAlgorithm, Tag(SupportedAlgorithm.CARE)],
        Annotated[N2NAlgorithm, Tag(SupportedAlgorithm.N2N)],
        Annotated[SegAlgorithm, Tag(SupportedAlgorithm.SEG)],
    ],
    Discriminator(_algo_disciminator),
]

# ------------------------ Validators --------------------------


def instantiate_config(config: dict[str, Any]) -> NGConfigs:
    """
    Instantiate a NG configuration from a configuration dictionary.

    This method uses a `TypeAdapter` to validate the configuration and instantiate the
    correct NG configuration.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary to validate.

    Returns
    -------
    NGConfigs
        Validated configuration as an NGConfig.

    Raises
    ------
    ValueError
        If the configuration is not valid.
    """
    adapter: TypeAdapter[NGConfigs] = TypeAdapter(NGConfigs)
    return adapter.validate_python(config)


def instantiate_algorithm_config(config: dict[str, Any]) -> NGAlgos:
    """
    Instantiate an algorithm configuration from a configuration dictionary.

    This method uses a `TypeAdapter` to validate the configuration and instantiate the
    correct algorithm configuration. Currently only compatible with UNet-based
    algorithms.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary to validate.

    Returns
    -------
    NGAlgos
        Validated configuration as one of the UNetBasedAlgorithm configurations.

    Raises
    ------
    ValueError
        If the configuration is not valid.
    """
    adapter: TypeAdapter[NGAlgos] = TypeAdapter(NGAlgos)
    return adapter.validate_python(config)


def instantiate_norm_config(config: dict[str, Any]) -> NormalizationConfig:
    """
    Instantiate a NormalizationConfig from a configuration dictionary.

    This method uses a `TypeAdapter` to validate the configuration and instantiate the
    correct NormalizationConfig.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary to validate.

    Returns
    -------
    NormalizationConfig
        Validated configuration as a NormalizationConfig.

    Raises
    ------
    ValueError
        If the configuration is not valid.
    """
    adapter: TypeAdapter[NormalizationConfig] = TypeAdapter(NormalizationConfig)
    return adapter.validate_python(config)
