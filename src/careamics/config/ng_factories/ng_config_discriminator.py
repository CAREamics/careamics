"""Discriminator for NG configurations."""

from typing import Annotated, Any, Union

from pydantic import Discriminator, Tag, TypeAdapter

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


# union
NGConfigs = Annotated[
    Union[
        Annotated[N2VConfiguration, Tag(SupportedAlgorithm.N2V)],
        Annotated[NGConfiguration, Tag(SupportedAlgorithm.CARE)],
        Annotated[NGConfiguration, Tag(SupportedAlgorithm.N2N)],
    ],
    Discriminator(_config_disciminator),
]


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
