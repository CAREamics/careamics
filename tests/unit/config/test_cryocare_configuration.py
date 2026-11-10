"""Tests for the CryoCARE top-level configuration."""

import pytest
from pydantic import ValidationError

from careamics.config import (
    CryoCAREAlgorithm,
    CryoCAREConfiguration,
    CryoCAREDataConfig,
)
from careamics.config.algorithms import CAREAlgorithm
from careamics.config.data.normalization_config import NoNormConfig
from careamics.config.factories.config_discriminators import instantiate_config


def _data_config() -> CryoCAREDataConfig:
    return CryoCAREDataConfig(
        mode="training",
        data_type="array",
        axes="ZYX",
        tilt_axis="Z",
        patching={"name": "stratified", "patch_size": (32, 32, 32)},
        normalization=NoNormConfig(),
        batch_size=1,
    )


def test_cryocare_configuration_accepts_only_cryocare_subconfigs() -> None:
    """Accept CryoCARE algorithm and data configurations."""
    config = CryoCAREConfiguration(
        experiment_name="cryocare",
        algorithm_config=CryoCAREAlgorithm(),
        data_config=_data_config(),
    )

    assert isinstance(config.algorithm_config, CryoCAREAlgorithm)
    assert isinstance(config.data_config, CryoCAREDataConfig)


def test_cryocare_configuration_rejects_other_algorithm() -> None:
    """Reject an algorithm configuration from another algorithm."""
    with pytest.raises(ValidationError):
        CryoCAREConfiguration(
            experiment_name="cryocare",
            algorithm_config=CAREAlgorithm(),
            data_config=_data_config(),
        )


def test_instantiate_config_returns_cryocare_configuration() -> None:
    """Instantiate the dedicated class through the configuration discriminator."""
    config = CryoCAREConfiguration(
        experiment_name="cryocare",
        algorithm_config=CryoCAREAlgorithm(),
        data_config=_data_config(),
    )

    restored = instantiate_config(config.model_dump())

    assert isinstance(restored, CryoCAREConfiguration)
