"""Discriminator for NG configurations."""

from typing import Annotated, Any, Union

from pydantic import Discriminator, Tag, TypeAdapter

from careamics.config.algorithms import CAREAlgorithm, N2NAlgorithm, N2VAlgorithm
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
    ],
    Discriminator(_config_disciminator),
]

NGAlgos = Annotated[
    Union[
        Annotated[N2VAlgorithm, Tag(SupportedAlgorithm.N2V)],
        Annotated[CAREAlgorithm, Tag(SupportedAlgorithm.CARE)],
        Annotated[N2NAlgorithm, Tag(SupportedAlgorithm.N2N)],
    ],
    Discriminator(_algo_disciminator),
]

# ------------------------ Validators --------------------------


def validate_ng_config(config: dict[str, Any]) -> NGConfigs:
    """
    Validate a configuration dictionary as an NGConfig.

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


def validate_ng_algos(config: dict[str, Any]) -> NGAlgos:
    """
    Validate a configuration dictionary as a UNetBasedAlgorithm.

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
