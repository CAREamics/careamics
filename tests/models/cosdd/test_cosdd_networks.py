"""Tests for the COSDD networks ported from the reference implementation."""

import math

import pytest
import torch

from careamics.config.architectures import COSDDConfig
from careamics.models.cosdd import (
    LadderVAE,
    PixelCNN,
    SDecoder,
    UNet,
    create_cosdd_networks,
    get_downsampling,
)
from careamics.models.cosdd.utils import (
    get_padded_size,
    interleave,
    spatial_pad_crop,
)


def _small_config(**kwargs) -> COSDDConfig:
    """Return a small COSDD configuration for fast tests."""
    params = {
        "architecture": "COSDD",
        "input_shape": (32, 32),
        "s_code_channels": 8,
        "n_layers": 4,
        "checkpointed": False,
    }
    params.update(kwargs)
    return COSDDConfig(**params)


@pytest.mark.parametrize(
    "input_shape, n_layers, expected",
    [
        # reference implementation defaults: 256x256 patches, 8 layers
        ((256, 256), 8, [[0, 0]] + [[1, 1]] * 7),
        # 16 can be halved 3 times, downsampling is removed from the first layer
        ((16, 16), 4, [[0, 0]] + [[1, 1]] * 3),
        # anisotropic 3D patches, Z can be halved once: downsampling is removed from
        # every other layer first, then from the remaining ones
        ((4, 32, 32), 4, [[0, 1, 1], [0, 1, 1], [0, 1, 1], [1, 1, 1]]),
    ],
)
def test_get_downsampling(input_shape, n_layers, expected):
    """Test the downsampling scheme against known values."""
    assert get_downsampling(input_shape, n_layers) == expected


@pytest.mark.parametrize("input_shape", [(2, 2), (8, 64), (16, 32, 256), (5, 33)])
@pytest.mark.parametrize("n_layers", [2, 4, 8, 14])
def test_get_downsampling_count(input_shape, n_layers):
    """Test that each axis is halved at most floor(log2(size)) - 1 times."""
    downsampling = get_downsampling(input_shape, n_layers)

    assert len(downsampling) == n_layers
    for axis, size in enumerate(input_shape):
        n_down = sum(layer[axis] for layer in downsampling)
        assert n_down == min(n_layers, math.floor(math.log2(size)) - 1)


@pytest.mark.parametrize("size", [(13, 32), (17, 9, 64)])
def test_spatial_pad_crop_roundtrip(size):
    """Test that padding to the downsampling size and cropping back is lossless."""
    x = torch.randn(1, 2, *size)
    padded_size = get_padded_size(size, [3] * len(size))

    padded = spatial_pad_crop(x, padded_size)
    assert all(s % 8 == 0 for s in padded.shape[2:])
    assert torch.equal(spatial_pad_crop(padded, size), x)


def test_interleave():
    """Test that interleaving alternates the channels of both tensors."""
    a = torch.zeros(1, 2, 4, 4)
    b = torch.ones(1, 2, 4, 4)

    out = interleave(a, b)
    assert out.shape == (1, 4, 4, 4)
    assert torch.equal(out[:, 0::2], a)
    assert torch.equal(out[:, 1::2], b)

    with pytest.raises(ValueError):
        interleave(a, torch.ones(1, 3, 4, 4))


@pytest.mark.parametrize(
    "input_shape, data_shape",
    [
        ((32, 32), (32, 32)),
        # prediction on a different, non power of 2, size
        ((32, 32), (40, 52)),
        ((8, 16, 16), (8, 16, 16)),
    ],
)
def test_lvae_forward(input_shape, data_shape):
    """Test the Ladder VAE output shapes and that its KL divergence is valid."""
    config = _small_config(input_shape=input_shape)
    vae = create_cosdd_networks(config).vae
    x = torch.randn(2, 1, *data_shape)

    out = vae(x)
    assert out["s_code"].shape == (2, config.s_code_channels, *data_shape)
    assert len(out["q_list"]) == len(out["p_list"]) == config.n_layers
    if input_shape == data_shape:
        kl = vae.kl_divergence(out["q_list"], out["p_list"])
        assert kl.ndim == 0
        assert kl >= 0


def test_lvae_sample_from_prior():
    """Test that prior samples have the shape of the training patches."""
    config = _small_config(input_shape=(16, 32))
    vae = create_cosdd_networks(config).vae

    s_code = vae.sample_from_prior(3)
    assert s_code.shape == (3, config.s_code_channels, 16, 32)


@pytest.mark.parametrize(
    "noise_direction, axis",
    [("x", -1), ("y", -2)],
)
def test_pixelcnn_is_causal(noise_direction, axis):
    """Test that the autoregressive decoder is causal along the noise direction.

    The parameters predicted for a pixel must only depend on the preceding pixels
    along the noise direction, not on the pixel itself or the following ones.
    """
    torch.manual_seed(0)
    ar_decoder = PixelCNN(
        colour_channels=1,
        s_code_channels=4,
        kernel_size=5,
        noise_direction=noise_direction,
        n_filters=8,
        n_layers=2,
        n_components=2,
    )
    x = torch.randn(1, 1, 16, 16)
    s_code = torch.randn(1, 4, 16, 16)
    position = 7

    x_perturbed = x.clone()
    x_perturbed.select(axis, position).add_(10.0)
    out = ar_decoder(x, s_code)
    out_perturbed = ar_decoder(x_perturbed, s_code)

    index = [slice(None)] * 4
    index[axis] = slice(0, position + 1)
    assert torch.equal(out[tuple(index)], out_perturbed[tuple(index)])
    index[axis] = slice(position + 1, None)
    assert not torch.equal(out[tuple(index)], out_perturbed[tuple(index)])


def test_pixelcnn_none_direction_ignores_noisy_input():
    """Test that pixel-independent noise only depends on the signal code."""
    ar_decoder = PixelCNN(
        colour_channels=1, s_code_channels=4, kernel_size=5, noise_direction="none"
    )
    s_code = torch.randn(1, 4, 8, 8)

    out = ar_decoder(torch.randn(1, 1, 8, 8), s_code)
    assert torch.equal(out, ar_decoder(torch.randn(1, 1, 8, 8), s_code))


@pytest.mark.parametrize("n_channels", [1, 2])
@pytest.mark.parametrize("noise_direction", ["x", "y", "none"])
def test_pixelcnn_loglikelihood_and_sample(n_channels, noise_direction):
    """Test the shapes of the log likelihood and of autoregressive samples."""
    ar_decoder = PixelCNN(
        colour_channels=n_channels,
        s_code_channels=4,
        kernel_size=5,
        noise_direction=noise_direction,
        n_filters=8,
        n_layers=2,
        n_components=3,
    )
    x = torch.randn(2, n_channels, 8, 12)
    s_code = torch.randn(2, 4, 8, 12)

    params = ar_decoder(x, s_code)
    assert params.shape == (2, 3 * 3 * n_channels, 8, 12)
    log_p = ar_decoder.loglikelihood(x, params)
    assert log_p.shape == x.shape
    assert torch.isfinite(log_p).all()

    if noise_direction != "none":
        sample = ar_decoder.sample(s_code)
        assert sample.shape == x.shape
        assert sample.device == s_code.device


def test_s_decoder():
    """Test the signal decoder output shape and minimum number of layers."""
    s_decoder = SDecoder(colour_channels=2, s_code_channels=4, n_filters=8)
    assert s_decoder(torch.randn(1, 4, 8, 8)).shape == (1, 2, 8, 8)

    with pytest.raises(ValueError):
        SDecoder(colour_channels=1, s_code_channels=4, n_layers=1)


@pytest.mark.parametrize("loss_fn", ["MSE", "L1"])
def test_direct_denoiser(loss_fn):
    """Test the direct denoiser output shape and loss."""
    direct_denoiser = UNet(
        colour_channels=1,
        downsampling=get_downsampling((16, 16), 2),
        n_filters=8,
        n_layers=2,
        loss_fn=loss_fn,
    )
    x = torch.randn(2, 1, 20, 12)
    y = direct_denoiser(x)
    assert y.shape == x.shape

    expected = (x - y).abs() if loss_fn == "L1" else (x - y) ** 2
    assert torch.allclose(direct_denoiser.loss(x, y), expected)

    with pytest.raises(ValueError):
        UNet(colour_channels=1, downsampling=[[1, 1]], n_layers=1, loss_fn="L2")


@pytest.mark.parametrize("use_direct_denoiser", [True, False])
def test_create_cosdd_networks(use_direct_denoiser):
    """Test that the factory builds the networks from the configuration."""
    config = _small_config(in_channels=2, use_direct_denoiser=use_direct_denoiser)

    networks = create_cosdd_networks(config)
    assert isinstance(networks.vae, LadderVAE)
    assert isinstance(networks.ar_decoder, PixelCNN)
    assert isinstance(networks.s_decoder, SDecoder)
    assert isinstance(networks.direct_denoiser, UNet) == use_direct_denoiser
    assert networks.ar_decoder.n_components == config.n_gaussians
