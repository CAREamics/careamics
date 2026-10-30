"""Tests for LVAE loss configuration classes."""

import pytest
from pydantic import ValidationError

from careamics.config.losses.loss_config import HDNLossConfig


def test_loss_types_are_fixed_per_subclass() -> None:
    """Each subclass fixes its own `loss_type`."""
    assert HDNLossConfig().loss_type == "hdn"


def test_invalid_loss_type_rejected() -> None:
    """A `loss_type` other than the subclass literal raises ValidationError."""
    with pytest.raises(ValidationError):
        HDNLossConfig(loss_type="microsplit")


def test_hdn_loss_config_has_no_split_weights() -> None:
    """HDN loss config exposes only the shared weights (no µSplit/denoiSplit)."""
    config = HDNLossConfig()
    assert config.loss_type == "hdn"
    assert config.reconstruction_weight == 1.0
    assert config.kl_weight == 1.0
    assert "gaussian_likelihood_weight" not in HDNLossConfig.model_fields
    assert "noise_model_likelihood_weight" not in HDNLossConfig.model_fields
