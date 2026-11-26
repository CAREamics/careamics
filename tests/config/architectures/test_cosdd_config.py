"""Tests for the COSDD architecture configuration."""

import pytest

from careamics.config.architectures import COSDDConfig


def test_defaults():
    """Test that the defaults follow the reference implementation."""
    config = COSDDConfig(architecture="COSDD")

    assert config.input_shape == (256, 256)
    assert config.s_code_channels == 64
    assert config.n_layers == 8
    assert config.n_gaussians == 3
    assert config.noise_direction == "x"
    assert config.use_direct_denoiser
    assert config.direct_denoiser_loss == "MSE"
    assert not config.scale_initialisation
    assert config.checkpointed
    assert config.z_dims == [32] * 8


@pytest.mark.parametrize("input_shape, is_3D", [((64, 64), False), ((8, 64, 64), True)])
def test_dimensions(input_shape, is_3D):
    """Test that the dimensionality is defined by the input shape."""
    config = COSDDConfig(architecture="COSDD", input_shape=input_shape)

    assert config.is_3D() == is_3D
    assert config.conv_dims == len(input_shape)


@pytest.mark.parametrize("input_shape", [(64,), (1, 8, 64, 64), (1, 64)])
def test_invalid_input_shape(input_shape):
    """Test that only 2D or 3D input shapes of size at least 2 are accepted."""
    with pytest.raises(ValueError):
        COSDDConfig(architecture="COSDD", input_shape=input_shape)


def test_z_noise_direction_requires_3D():
    """Test that noise correlated along Z requires a 3D model."""
    with pytest.raises(ValueError, match="3D"):
        COSDDConfig(architecture="COSDD", input_shape=(64, 64), noise_direction="z")

    config = COSDDConfig(
        architecture="COSDD", input_shape=(8, 64, 64), noise_direction="z"
    )
    assert config.noise_direction == "z"


def test_channels():
    """Test that COSDD outputs as many channels as its input."""
    config = COSDDConfig(architecture="COSDD", in_channels=3)

    assert config.get_num_input_channels() == 3
    assert config.get_num_output_channels() == 3
    assert config.uses_batch_norm()


@pytest.mark.parametrize(
    "params",
    [{"n_layers": 1}, {"s_code_channels": 1}, {"n_gaussians": 0}, {"in_channels": 0}],
)
def test_invalid_parameters(params):
    """Test that invalid network parameters are rejected."""
    with pytest.raises(ValueError):
        COSDDConfig(architecture="COSDD", **params)
