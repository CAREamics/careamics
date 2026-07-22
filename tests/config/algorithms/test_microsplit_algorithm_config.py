# TODO - revisit: tests do not follow the new test organization and style
import warnings
from pathlib import Path

import numpy as np
import pytest

from careamics.config.algorithms import MicroSplitAlgorithm
from careamics.config.architectures import LVAEConfig
from careamics.config.losses.loss_config import LVAELossConfig
from careamics.config.noise_model.noise_model_config import (
    GaussianMixtureNMConfig,
    MultiChannelNMConfig,
)


def _dummy_noise_model(
    tmp_path: Path, dummy_noise_model: dict, n_channels: int = 1
) -> MultiChannelNMConfig:
    """Create a multi-channel noise model configuration from dummy weights."""
    np.savez(tmp_path / "dummy_noise_model.npz", **dummy_noise_model)
    gmm = GaussianMixtureNMConfig(
        model_type="GaussianMixtureNoiseModel",
        path=tmp_path / "dummy_noise_model.npz",
    )
    return MultiChannelNMConfig(noise_models=[gmm] * n_channels)


def test_instantiation(tmp_path: Path, create_dummy_noise_model):
    """Test that the MicroSplit algorithm can be instantiated correctly."""
    loss = LVAELossConfig(
        loss_type="microsplit", denoisplit_weight=0.9, musplit_weight=0.1
    )
    config = MicroSplitAlgorithm(
        loss=loss,
        model=LVAEConfig(architecture="LVAE"),
        noise_model=_dummy_noise_model(tmp_path, create_dummy_noise_model),
    )
    assert config.algorithm == "microsplit"
    assert config.model.architecture == "LVAE"
    assert config.noise_model is not None
    assert config.is_supervised


def test_wrong_algorithm(tmp_path: Path, create_dummy_noise_model):
    """Test that another algorithm name is rejected."""
    with pytest.raises(ValueError):
        MicroSplitAlgorithm(
            algorithm="hdn",
            model=LVAEConfig(architecture="LVAE"),
            noise_model=_dummy_noise_model(tmp_path, create_dummy_noise_model),
        )


def test_wrong_loss_type(tmp_path: Path, create_dummy_noise_model):
    """Test that a non-MicroSplit loss is rejected."""
    with pytest.raises(ValueError):
        MicroSplitAlgorithm(
            loss=LVAELossConfig(loss_type="hdn"),
            model=LVAEConfig(architecture="LVAE"),
            noise_model=_dummy_noise_model(tmp_path, create_dummy_noise_model),
        )


def test_missing_noise_model_warns():
    """Test that denoisplit_weight > 0 without a noise model warns."""
    loss = LVAELossConfig(
        loss_type="microsplit", denoisplit_weight=0.9, musplit_weight=0.1
    )
    with pytest.warns(UserWarning, match="noise model is required"):
        MicroSplitAlgorithm(loss=loss, model=LVAEConfig(architecture="LVAE"))


def test_no_warning_without_denoisplit():
    """Test that no warning is raised when the noise model likelihood is disabled."""
    loss = LVAELossConfig(
        loss_type="microsplit", denoisplit_weight=0.0, musplit_weight=1.0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        MicroSplitAlgorithm(loss=loss, model=LVAEConfig(architecture="LVAE"))


def test_noise_model_count_mismatch(tmp_path: Path, create_dummy_noise_model):
    """Test that the number of noise models must match the output channels."""
    with pytest.raises(ValueError):
        MicroSplitAlgorithm(
            model=LVAEConfig(architecture="LVAE", output_channels=2),
            noise_model=_dummy_noise_model(
                tmp_path, create_dummy_noise_model, n_channels=1
            ),
        )


def test_predict_logvar_mismatch():
    """Test that model and loss `predict_logvar` must match."""
    loss = LVAELossConfig(
        loss_type="microsplit",
        denoisplit_weight=0.0,
        musplit_weight=1.0,
        predict_logvar=True,
    )
    with pytest.raises(ValueError):
        MicroSplitAlgorithm(
            loss=loss,
            model=LVAEConfig(architecture="LVAE", predict_logvar=False),
        )
