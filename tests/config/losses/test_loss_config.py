"""Tests for LVAE loss configuration classes."""

from typing import Literal

import pytest
from pydantic import ValidationError

from careamics.config.losses.loss_config import KLLossConfig, LVAELossConfig


def test_kl_loss_config_default_values() -> None:
    """Test that default values are correctly set."""
    config = KLLossConfig()
    assert config.current_epoch == 0


@pytest.mark.parametrize("current_epoch", [0, 1, 50, 100])
def test_kl_loss_config_current_epoch(current_epoch: int) -> None:
    """Test that various current_epoch values are accepted."""
    config = KLLossConfig(current_epoch=current_epoch)
    assert config.current_epoch == current_epoch


def test_kl_loss_config_with_current_epoch() -> None:
    """Test configuration with custom current_epoch."""
    config = KLLossConfig(current_epoch=10)
    assert config.current_epoch == 10


@pytest.mark.parametrize(
    "loss_type",
    ["hdn", "microsplit", "musplit", "denoisplit", "denoisplit_musplit"],
)
def test_lvae_loss_config_valid_loss_type(
    loss_type: Literal[
        "hdn", "microsplit", "musplit", "denoisplit", "denoisplit_musplit"
    ],
) -> None:
    """Test that valid loss_type values are accepted."""
    config = LVAELossConfig(loss_type=loss_type)
    assert config.loss_type == loss_type


def test_lvae_loss_config_invalid_loss_type() -> None:
    """Test that invalid loss_type raises ValidationError."""
    with pytest.raises(ValidationError):
        LVAELossConfig(loss_type="invalid_loss")


def test_lvae_loss_config_default_values() -> None:
    """Test that default values are correctly set."""
    config = LVAELossConfig(loss_type="microsplit")
    assert config.loss_type == "microsplit"
    assert config.reconstruction_weight == 1.0
    assert config.kl_weight == 1.0
    assert config.musplit_weight == 0.0
    assert config.denoisplit_weight == 1.0
    assert isinstance(config.kl_params, KLLossConfig)


@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0, 2.0])
def test_lvae_loss_config_reconstruction_weight(weight: float) -> None:
    """Test that various reconstruction_weight values are accepted."""
    config = LVAELossConfig(loss_type="microsplit", reconstruction_weight=weight)
    assert config.reconstruction_weight == weight


@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0, 2.0])
def test_lvae_loss_config_kl_weight(weight: float) -> None:
    """Test that various kl_weight values are accepted."""
    config = LVAELossConfig(loss_type="microsplit", kl_weight=weight)
    assert config.kl_weight == weight


@pytest.mark.parametrize("weight", [0.0, 0.1, 0.5, 1.0])
def test_lvae_loss_config_musplit_weight(weight: float) -> None:
    """Test that various musplit_weight values are accepted."""
    config = LVAELossConfig(loss_type="microsplit", musplit_weight=weight)
    assert config.musplit_weight == weight


@pytest.mark.parametrize("weight", [0.0, 0.5, 0.9, 1.0])
def test_lvae_loss_config_denoisplit_weight(weight: float) -> None:
    """Test that various denoisplit_weight values are accepted."""
    config = LVAELossConfig(loss_type="microsplit", denoisplit_weight=weight)
    assert config.denoisplit_weight == weight


@pytest.mark.parametrize("weight", [-1.0, -0.5])
def test_lvae_loss_config_negative_weights(weight: float) -> None:
    """Test that negative weight values are accepted (no constraint enforced)."""
    config = LVAELossConfig(loss_type="microsplit", reconstruction_weight=weight)
    assert config.reconstruction_weight == weight


def test_lvae_loss_config_weight_combinations() -> None:
    """Test various weight combinations for microsplit loss."""
    # muSplit only (gaussian_weight=1, nm_weight=0)
    config1 = LVAELossConfig(
        loss_type="microsplit", musplit_weight=1.0, denoisplit_weight=0.0
    )
    assert config1.musplit_weight == 1.0
    assert config1.denoisplit_weight == 0.0

    # denoiSplit only (gaussian_weight=0, nm_weight=1)
    config2 = LVAELossConfig(
        loss_type="microsplit", musplit_weight=0.0, denoisplit_weight=1.0
    )
    assert config2.musplit_weight == 0.0
    assert config2.denoisplit_weight == 1.0

    # Balanced (0.5, 0.5)
    config3 = LVAELossConfig(
        loss_type="microsplit", musplit_weight=0.5, denoisplit_weight=0.5
    )
    assert config3.musplit_weight == 0.5
    assert config3.denoisplit_weight == 0.5


def test_lvae_loss_config_kl_params_integration() -> None:
    """Test that KLLossConfig is properly integrated."""
    kl_config = KLLossConfig(current_epoch=5)
    config = LVAELossConfig(loss_type="microsplit", kl_params=kl_config)
    assert config.kl_params.current_epoch == 5


def test_lvae_loss_config_kl_params_default() -> None:
    """Test that KLLossConfig defaults are used when not specified."""
    config = LVAELossConfig(loss_type="microsplit")
    assert config.kl_params.current_epoch == 0


def test_lvae_loss_config_hdn_loss_type() -> None:
    """Test HDN loss type configuration."""
    config = LVAELossConfig(loss_type="hdn")
    assert config.loss_type == "hdn"
    # HDN should work with default weights
    assert config.reconstruction_weight == 1.0
    assert config.kl_weight == 1.0




def test_lvae_loss_config_full_config() -> None:
    """Test complete configuration with all parameters."""
    kl_config = KLLossConfig(current_epoch=10)
    config = LVAELossConfig(
        loss_type="microsplit",
        reconstruction_weight=1.5,
        kl_weight=0.8,
        musplit_weight=0.3,
        denoisplit_weight=0.7,
        kl_params=kl_config,
    )
    assert config.loss_type == "microsplit"
    assert config.reconstruction_weight == 1.5
    assert config.kl_weight == 0.8
    assert config.musplit_weight == 0.3
    assert config.denoisplit_weight == 0.7
    assert config.kl_params.current_epoch == 10


def test_lvae_loss_config_modification() -> None:
    """Test that config values can be modified after creation."""
    config = LVAELossConfig(loss_type="microsplit")
    # Modify values
    config.musplit_weight = 0.8
    config.denoisplit_weight = 0.2
    config.reconstruction_weight = 2.0
    # Verify modifications
    assert config.musplit_weight == 0.8
    assert config.denoisplit_weight == 0.2
    assert config.reconstruction_weight == 2.0


def test_lvae_loss_config_kl_params_modification() -> None:
    """Test that nested KL params can be modified."""
    config = LVAELossConfig(loss_type="microsplit")
    # Modify KL params
    config.kl_params.current_epoch = 25
    # Verify modifications
    assert config.kl_params.current_epoch == 25

