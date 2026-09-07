"""Tests for LVAE loss configuration classes."""

import pytest
from pydantic import ValidationError

from careamics.config.losses.loss_config import (
    HDNLossConfig,
    MicroSplitLossConfig,
)


def test_loss_types_are_fixed_per_subclass() -> None:
    """Each subclass fixes its own `loss_type`."""
    assert HDNLossConfig().loss_type == "hdn"
    assert MicroSplitLossConfig().loss_type == "microsplit"


def test_invalid_loss_type_rejected() -> None:
    """A `loss_type` other than the subclass literal raises ValidationError."""
    with pytest.raises(ValidationError):
        MicroSplitLossConfig(loss_type="invalid_loss")
    with pytest.raises(ValidationError):
        HDNLossConfig(loss_type="microsplit")


def test_microsplit_loss_config_default_values() -> None:
    """Test that default values are correctly set."""
    config = MicroSplitLossConfig()
    assert config.loss_type == "microsplit"
    assert config.reconstruction_weight == 1.0
    assert config.kl_weight == 1.0
    assert config.gaussian_likelihood_weight == 0.1
    assert config.noise_model_likelihood_weight == 0.9


def test_hdn_loss_config_has_no_split_weights() -> None:
    """HDN loss config exposes only the shared weights (no µSplit/denoiSplit)."""
    config = HDNLossConfig()
    assert config.loss_type == "hdn"
    assert config.reconstruction_weight == 1.0
    assert config.kl_weight == 1.0
    assert "gaussian_likelihood_weight" not in HDNLossConfig.model_fields
    assert "noise_model_likelihood_weight" not in HDNLossConfig.model_fields


@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0, 2.0])
def test_reconstruction_weight(weight: float) -> None:
    """Test that various reconstruction_weight values are accepted."""
    config = MicroSplitLossConfig(reconstruction_weight=weight)
    assert config.reconstruction_weight == weight


@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0, 2.0])
def test_kl_weight(weight: float) -> None:
    """Test that various kl_weight values are accepted."""
    config = MicroSplitLossConfig(kl_weight=weight)
    assert config.kl_weight == weight


@pytest.mark.parametrize("weight", [0.0, 0.1, 0.5, 1.0])
def test_gaussian_likelihood_weight(weight: float) -> None:
    """Test that various gaussian_likelihood_weight values are accepted."""
    config = MicroSplitLossConfig(gaussian_likelihood_weight=weight)
    assert config.gaussian_likelihood_weight == weight


@pytest.mark.parametrize("weight", [0.0, 0.5, 0.9, 1.0])
def test_noise_model_likelihood_weight(weight: float) -> None:
    """Test that various noise_model_likelihood_weight values are accepted."""
    config = MicroSplitLossConfig(noise_model_likelihood_weight=weight)
    assert config.noise_model_likelihood_weight == weight


@pytest.mark.parametrize("weight", [-1.0, -0.5])
def test_negative_weights_rejected(weight: float) -> None:
    """Test that negative weight values are rejected."""
    with pytest.raises(ValidationError):
        MicroSplitLossConfig(reconstruction_weight=weight)
    with pytest.raises(ValidationError):
        MicroSplitLossConfig(gaussian_likelihood_weight=weight)
    with pytest.raises(ValidationError):
        MicroSplitLossConfig(noise_model_likelihood_weight=weight)


def test_weight_combinations() -> None:
    """Test various weight combinations for microsplit loss."""
    # muSplit only
    config1 = MicroSplitLossConfig(
        gaussian_likelihood_weight=1.0, noise_model_likelihood_weight=0.0
    )
    assert config1.gaussian_likelihood_weight == 1.0
    assert config1.noise_model_likelihood_weight == 0.0

    # denoiSplit only
    config2 = MicroSplitLossConfig(
        gaussian_likelihood_weight=0.0, noise_model_likelihood_weight=1.0
    )
    assert config2.gaussian_likelihood_weight == 0.0
    assert config2.noise_model_likelihood_weight == 1.0

    # Balanced
    config3 = MicroSplitLossConfig(
        gaussian_likelihood_weight=0.5, noise_model_likelihood_weight=0.5
    )
    assert config3.gaussian_likelihood_weight == 0.5
    assert config3.noise_model_likelihood_weight == 0.5


def test_full_config() -> None:
    """Test complete configuration with all parameters."""
    config = MicroSplitLossConfig(
        reconstruction_weight=1.5,
        kl_weight=0.8,
        gaussian_likelihood_weight=0.3,
        noise_model_likelihood_weight=0.7,
    )
    assert config.loss_type == "microsplit"
    assert config.reconstruction_weight == 1.5
    assert config.kl_weight == 0.8
    assert config.gaussian_likelihood_weight == 0.3
    assert config.noise_model_likelihood_weight == 0.7


def test_config_modification() -> None:
    """Test that config values can be modified after creation."""
    config = MicroSplitLossConfig()
    config.gaussian_likelihood_weight = 0.8
    config.noise_model_likelihood_weight = 0.2
    config.reconstruction_weight = 2.0
    assert config.gaussian_likelihood_weight == 0.8
    assert config.noise_model_likelihood_weight == 0.2
    assert config.reconstruction_weight == 2.0
