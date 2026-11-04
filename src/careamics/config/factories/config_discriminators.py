"""Discriminator for NG configurations."""

from typing import Annotated, Any, Union

from pydantic import Discriminator, Tag, TypeAdapter

from careamics.config.algorithms import (
    CAREAlgorithm,
    HDNAlgorithm,
    MicroSplitAlgorithm,
    N2NAlgorithm,
    N2VAlgorithm,
)
from careamics.config.configuration import Configuration
from careamics.config.data.data_config import DataConfig
from careamics.config.data.microsplit_data_config import MicroSplitDataConfig
from careamics.config.data.normalization_config import NormalizationConfig
from careamics.config.hdn_configuration import HDNConfiguration
from careamics.config.microsplit_configuration import MicroSplitConfiguration
from careamics.config.n2v_configuration import N2VConfiguration
from careamics.config.support import SupportedAlgorithm

OTHERS = "others"

_MICROSPLIT_DATA_FIELDS = frozenset(MicroSplitDataConfig.model_fields) - frozenset(
    DataConfig.model_fields
)


def _config_discriminator(v: Any) -> SupportedAlgorithm | None:
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


def _algo_discriminator(algo: Any) -> SupportedAlgorithm | None:
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


def _data_discriminator(data: Any) -> str | None:
    """
    Extract data configuration type from dict for Pydantic discriminator.

    Parameters
    ----------
    data : Any
        Data configuration dictionary.

    Returns
    -------
    str or None
        MicroSplit if any MicroSplit-only field is present, `OTHERS` otherwise, and
        None if `data` is not a dictionary.
    """
    if not isinstance(data, dict):
        return None
    if _MICROSPLIT_DATA_FIELDS.intersection(data):
        return SupportedAlgorithm.MICROSPLIT
    return OTHERS


# ------------------------ Unions --------------------------

Config = Annotated[
    Union[
        Annotated[N2VConfiguration, Tag(SupportedAlgorithm.N2V)],
        Annotated[Configuration, Tag(SupportedAlgorithm.CARE)],
        Annotated[Configuration, Tag(SupportedAlgorithm.N2N)],
        Annotated[HDNConfiguration, Tag(SupportedAlgorithm.HDN)],
        Annotated[MicroSplitConfiguration, Tag(SupportedAlgorithm.MICROSPLIT)],
    ],
    Discriminator(_config_discriminator),
]

AlgorithmConfig = Annotated[
    Union[
        Annotated[N2VAlgorithm, Tag(SupportedAlgorithm.N2V)],
        Annotated[CAREAlgorithm, Tag(SupportedAlgorithm.CARE)],
        Annotated[N2NAlgorithm, Tag(SupportedAlgorithm.N2N)],
        Annotated[HDNAlgorithm, Tag(SupportedAlgorithm.HDN)],
        Annotated[MicroSplitAlgorithm, Tag(SupportedAlgorithm.MICROSPLIT)],
    ],
    Discriminator(_algo_discriminator),
]

DataCfg = Annotated[
    Union[
        Annotated[MicroSplitDataConfig, Tag(SupportedAlgorithm.MICROSPLIT)],
        Annotated[DataConfig, Tag(OTHERS)],
    ],
    Discriminator(_data_discriminator),
]

# ------------------------ Validators --------------------------


def instantiate_config(config: dict[str, Any]) -> Config:
    """
    Instantiate a configuration from a configuration dictionary.

    This method uses a `TypeAdapter` to validate the configuration and instantiate the
    correct configuration.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary to validate.

    Returns
    -------
    Config
        Validated configuration as a Config.

    Raises
    ------
    ValueError
        If the configuration is not valid.
    """
    adapter: TypeAdapter[Config] = TypeAdapter(Config)
    return adapter.validate_python(config)


def instantiate_algorithm_config(config: dict[str, Any]) -> AlgorithmConfig:
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
    AlgorithmConfig
        Validated configuration as one of the UNetBasedAlgorithm configurations.

    Raises
    ------
    ValueError
        If the configuration is not valid.
    """
    adapter: TypeAdapter[AlgorithmConfig] = TypeAdapter(AlgorithmConfig)
    return adapter.validate_python(config)


def instantiate_data_config(config: dict[str, Any]) -> DataCfg:
    """
    Instantiate a data configuration from a configuration dictionary.

    This method uses a `TypeAdapter` to validate the configuration and instantiate the
    correct data configuration.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary to validate.

    Returns
    -------
    DataCfg
        Validated configuration as a `MicroSplitDataConfig` or a `DataConfig`.

    Raises
    ------
    ValueError
        If the configuration is not valid.
    """
    adapter: TypeAdapter[DataCfg] = TypeAdapter(DataCfg)
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
