# TODO - revisit: tests do not follow the new test organization and style
import warnings

import pytest

from careamics.config.algorithms import MicroSplitAlgorithm
from careamics.config.architectures import LVAEConfig
from careamics.config.losses.loss_config import (
    HDNLossConfig,
    MicroSplitLossConfig,
)


# NOTE: the noise model is no longer part of the configuration; it is injected at
# training time (`CAREamist.train(noise_model=...)` / `MicroSplitModule.set_noise_model`).
# Noise-model presence / channel-count / unused-model checks are covered by the module
# tests (see tests for `set_noise_model` and `on_fit_start`), not here.


def test_instantiation():
    """Test that the MicroSplit algorithm can be instantiated correctly."""
    loss = MicroSplitLossConfig(
        noise_model_likelihood_weight=0.9, gaussian_likelihood_weight=0.1
    )
    config = MicroSplitAlgorithm(loss=loss, model=LVAEConfig(architecture="LVAE"))
    assert config.algorithm == "microsplit"
    assert config.model.architecture == "LVAE"
    assert config.is_supervised()


def test_no_noise_model_field():
    """The noise model is not part of the configuration."""
    assert "noise_model" not in MicroSplitAlgorithm.model_fields


def test_wrong_algorithm():
    """Test that another algorithm name is rejected."""
    with pytest.raises(ValueError):
        MicroSplitAlgorithm(
            algorithm="hdn",
            model=LVAEConfig(architecture="LVAE"),
        )


def test_wrong_loss_type():
    """Test that a non-MicroSplit loss is rejected."""
    with pytest.raises(ValueError):
        MicroSplitAlgorithm(
            loss=HDNLossConfig(),
            model=LVAEConfig(architecture="LVAE"),
        )


def test_no_warning_at_config_time():
    """No noise-model warning is emitted at configuration time.

    The noise model is provided at training time, so the config validators must not
    warn about a missing / unused noise model.
    """
    loss = MicroSplitLossConfig(
        noise_model_likelihood_weight=0.9, gaussian_likelihood_weight=0.1
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        MicroSplitAlgorithm(loss=loss, model=LVAEConfig(architecture="LVAE"))


def test_predict_logvar_mismatch():
    """Test that model and loss `predict_logvar` must match."""
    loss = MicroSplitLossConfig(
        noise_model_likelihood_weight=0.0,
        gaussian_likelihood_weight=1.0,
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


def test_z_stride_must_be_one():
    """Test that the Z (depth) stride must be 1 for 3D LVAE models."""
    with pytest.raises(ValueError, match=r"Z \(depth\) stride must be 1"):
        MicroSplitAlgorithm(
            model=LVAEConfig(
                architecture="LVAE",
                input_shape=(64, 64, 64),
                encoder_conv_strides=[2, 2, 2],
                decoder_conv_strides=[2, 2, 2],
                z_dims=[128, 128],
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
            loss=MicroSplitLossConfig(
                gaussian_likelihood_weight=0.0, noise_model_likelihood_weight=0.0
            ),
            model=LVAEConfig(architecture="LVAE"),
        )


def test_predict_logvar_required_for_musplit():
    """Test that predict_logvar must be True when the muSplit likelihood is active."""
    with pytest.raises(ValueError, match="must be True when the muSplit"):
        MicroSplitAlgorithm(
            loss=MicroSplitLossConfig(
                gaussian_likelihood_weight=0.5,
                noise_model_likelihood_weight=0.5,
                predict_logvar=False,
            ),
            model=LVAEConfig(architecture="LVAE", predict_logvar=False),
        )
