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
    assert config.is_supervised()


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


def test_mmse_count_lower_bound():
    """Test that `mmse_count` must be at least 1."""
    with pytest.raises(ValueError):
        MicroSplitAlgorithm(model=LVAEConfig(architecture="LVAE"), mmse_count=0)


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


def test_spatial_shape_divisibility():
    """Test that input dims must be divisible by stride ** len(z_dims)."""
    # 72 is not divisible by 2 ** 4 (default z_dims has length 4)
    with pytest.raises(ValueError, match="must be divisible"):
        MicroSplitAlgorithm(model=LVAEConfig(architecture="LVAE", input_shape=(64, 72)))


def test_odd_depth_for_2d_decoder():
    """Test that a 3D encoder with a 2D decoder requires an odd depth."""
    with pytest.raises(ValueError, match="odd depth"):
        MicroSplitAlgorithm(
            model=LVAEConfig(
                architecture="LVAE",
                input_shape=(8, 64, 64),
                encoder_conv_strides=[1, 2, 2],
                decoder_conv_strides=[2, 2],
            )
        )


def test_conv_strides_dim_mismatch():
    """Test that encoder strides length must match the input dimensionality."""
    with pytest.raises(ValueError, match="number of encoder conv strides"):
        MicroSplitAlgorithm(
            model=LVAEConfig(
                architecture="LVAE",
                input_shape=(64, 64),
                encoder_conv_strides=[2, 2, 2],
            )
        )


def test_multiscale_count_out_of_range():
    """Test that multiscale count cannot exceed len(z_dims) + 1."""
    with pytest.raises(ValueError, match="Multiscale count"):
        MicroSplitAlgorithm(
            model=LVAEConfig(architecture="LVAE", z_dims=[128, 128], multiscale_count=5)
        )


def test_requires_at_least_one_likelihood():
    """Test that both likelihood weights cannot be zero."""
    with pytest.raises(ValueError, match="At least one"):
        MicroSplitAlgorithm(
            loss=LVAELossConfig(
                loss_type="microsplit", musplit_weight=0.0, denoisplit_weight=0.0
            ),
            model=LVAEConfig(architecture="LVAE"),
        )


def test_predict_logvar_required_for_musplit():
    """Test that predict_logvar must be True when the muSplit likelihood is active."""
    with pytest.raises(ValueError, match="must be True when the muSplit"):
        MicroSplitAlgorithm(
            loss=LVAELossConfig(
                loss_type="microsplit",
                musplit_weight=0.5,
                denoisplit_weight=0.5,
                predict_logvar=False,
            ),
            model=LVAEConfig(architecture="LVAE", predict_logvar=False),
        )


def test_mmse_count_must_be_positive():
    """Test that mmse_count must be at least 1."""
    with pytest.raises(ValueError):
        MicroSplitAlgorithm(model=LVAEConfig(architecture="LVAE"), mmse_count=0)


def test_unused_noise_model_warns(tmp_path: Path, create_dummy_noise_model):
    """Test that a noise model with denoisplit_weight=0 warns it will not be used."""
    loss = LVAELossConfig(
        loss_type="microsplit", musplit_weight=1.0, denoisplit_weight=0.0
    )
    with pytest.warns(UserWarning, match="will not be used"):
        MicroSplitAlgorithm(
            loss=loss,
            model=LVAEConfig(architecture="LVAE"),
            noise_model=_dummy_noise_model(tmp_path, create_dummy_noise_model),
        )
