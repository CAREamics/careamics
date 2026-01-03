"""Tests for FCN (Fully Convolutional Network) losses."""

from contextlib import nullcontext as does_not_raise
from typing import Callable, Union

import pytest
import torch

from careamics.config.support import SupportedActivation
from careamics.losses.fcn.losses import (
    mae_loss,
    mse_loss,
    n2v_loss,
    n2v_poisson_loss,
)
from careamics.losses.loss_factory import SupportedLoss, loss_factory
from careamics.models.activation import get_activation


@pytest.mark.parametrize(
    "loss_type, exp_loss_func, exp_error",
    [
        (SupportedLoss.N2V, n2v_loss, does_not_raise()),
        (SupportedLoss.N2V_POISSON, n2v_poisson_loss, does_not_raise()),
        (SupportedLoss.MSE, mse_loss, does_not_raise()),
        (SupportedLoss.MAE, mae_loss, does_not_raise()),
        ("n2v", n2v_loss, does_not_raise()),
        ("n2v_poisson", n2v_poisson_loss, does_not_raise()),
        ("mse", mse_loss, does_not_raise()),
        ("mae", mae_loss, does_not_raise()),
    ],
)
def test_fcn_loss_factory(
    loss_type: Union[SupportedLoss, str], exp_loss_func: Callable, exp_error: Callable
):
    """Test that loss_factory returns correct FCN loss functions."""
    with exp_error:
        loss_func = loss_factory(loss_type)
        assert loss_func is not None
        assert callable(loss_func)
        assert loss_func == exp_loss_func


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("n_channels", [1, 2])
@pytest.mark.parametrize("img_size", [32, 64])
def test_n2v_loss(batch_size: int, n_channels: int, img_size: int):
    """Test N2V loss computation."""
    # Create test data
    manipulated = torch.rand(batch_size, n_channels, img_size, img_size)
    original = torch.rand(batch_size, n_channels, img_size, img_size)
    mask = torch.randint(0, 2, (batch_size, n_channels, img_size, img_size))

    # Compute loss
    loss = n2v_loss(manipulated, original, mask)

    # Check output
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar
    assert loss >= 0  # MSE loss is non-negative


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("n_channels", [1, 2])
@pytest.mark.parametrize("img_size", [32, 64])
def test_n2v_poisson_loss(batch_size: int, n_channels: int, img_size: int):
    """Test N2V Poisson loss computation."""
    # Create test data
    # Predictions should be positive (photon rates)
    manipulated = torch.rand(batch_size, n_channels, img_size, img_size) * 100
    # Original should be counts (non-negative integers, but we use floats)
    original = torch.randint(0, 200, (batch_size, n_channels, img_size, img_size)).float()
    mask = torch.randint(0, 2, (batch_size, n_channels, img_size, img_size))

    # Compute loss
    loss = n2v_poisson_loss(manipulated, original, mask)

    # Check output
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar
    assert torch.isfinite(loss)  # should not be inf or nan


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("img_size", [32, 64])
def test_n2v_poisson_loss_with_zero_predictions(batch_size: int, img_size: int):
    """Test that N2V Poisson loss handles zero/near-zero predictions safely."""
    # Create test data with some zero predictions
    manipulated = torch.zeros(batch_size, 1, img_size, img_size)
    original = torch.randint(0, 10, (batch_size, 1, img_size, img_size)).float()
    mask = torch.ones(batch_size, 1, img_size, img_size)

    # Compute loss - should not raise error due to clamping
    loss = n2v_poisson_loss(manipulated, original, mask)

    # Check output
    assert isinstance(loss, torch.Tensor)
    assert torch.isfinite(loss)  # should not be inf or nan


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("out_channels", [1, 2])
@pytest.mark.parametrize("in_channels", [2, 3])
def test_n2v_loss_multi_channel(batch_size: int, out_channels: int, in_channels: int):
    """Test N2V loss with different input/output channel counts."""
    img_size = 32

    # Output has fewer channels than input
    manipulated = torch.rand(batch_size, out_channels, img_size, img_size)
    original = torch.rand(batch_size, in_channels, img_size, img_size)

    # Only some channels have masks (data channels)
    mask = torch.zeros(batch_size, in_channels, img_size, img_size)
    mask[:, :out_channels, :, :] = torch.randint(0, 2, (batch_size, out_channels, img_size, img_size))

    # Compute loss
    loss = n2v_loss(manipulated, original, mask)

    # Check output
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss >= 0


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("out_channels", [1, 2])
@pytest.mark.parametrize("in_channels", [2, 3])
def test_n2v_poisson_loss_multi_channel(batch_size: int, out_channels: int, in_channels: int):
    """Test N2V Poisson loss with different input/output channel counts."""
    img_size = 32

    # Output has fewer channels than input
    manipulated = torch.rand(batch_size, out_channels, img_size, img_size) * 100
    original = torch.randint(0, 200, (batch_size, in_channels, img_size, img_size)).float()

    # Only some channels have masks (data channels)
    mask = torch.zeros(batch_size, in_channels, img_size, img_size)
    mask[:, :out_channels, :, :] = torch.randint(0, 2, (batch_size, out_channels, img_size, img_size))

    # Compute loss
    loss = n2v_poisson_loss(manipulated, original, mask)

    # Check output
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("n_channels", [1, 2])
@pytest.mark.parametrize("img_size", [32, 64])
def test_mse_loss(batch_size: int, n_channels: int, img_size: int):
    """Test MSE loss computation."""
    # Create test data
    source = torch.rand(batch_size, n_channels, img_size, img_size)
    target = torch.rand(batch_size, n_channels, img_size, img_size)

    # Compute loss
    loss = mse_loss(source, target)

    # Check output
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar
    assert loss >= 0  # MSE is non-negative


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("n_channels", [1, 2])
@pytest.mark.parametrize("img_size", [32, 64])
def test_mae_loss(batch_size: int, n_channels: int, img_size: int):
    """Test MAE loss computation."""
    # Create test data
    source = torch.rand(batch_size, n_channels, img_size, img_size)
    target = torch.rand(batch_size, n_channels, img_size, img_size)

    # Compute loss
    loss = mae_loss(source, target)

    # Check output
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar
    assert loss >= 0  # MAE is non-negative


def test_softplus_activation():
    """Test that Softplus activation is properly registered and outputs positive values."""
    # Get the activation
    activation = get_activation(SupportedActivation.SOFTPLUS)
    assert activation is not None

    # Test that it produces positive outputs
    batch_size, channels, img_size = 4, 2, 32
    x = torch.randn(batch_size, channels, img_size, img_size)  # can be negative
    output = activation(x)

    # Check all outputs are positive
    assert torch.all(output > 0), "Softplus should produce strictly positive outputs"

    # Test that it's smooth (no zeros like ReLU would produce)
    x_negative = torch.tensor([-10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0])
    output_negative = activation(x_negative)
    assert torch.all(output_negative > 0), "Softplus should be positive even for negative inputs"
