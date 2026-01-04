"""Tests for 1D UNet."""

import pytest
import torch

from careamics.config.architectures import UNetConfig
from careamics.models.unet import UNet


class TestUNet1D:
    """Test UNet with 1D data."""

    @pytest.mark.parametrize("depth", [2, 3, 4])
    @pytest.mark.parametrize("num_channels_init", [32, 64])
    def test_unet1d_creation(self, depth: int, num_channels_init: int):
        """Test 1D UNet can be created with different configurations."""
        config = UNetConfig(
            architecture="UNet",
            conv_dims=1,
            in_channels=1,
            num_classes=1,
            depth=depth,
            num_channels_init=num_channels_init,
        )
        model = UNet(**config.model_dump())
        assert model is not None
        assert model.depth == depth

    def test_unet1d_forward_pass(self):
        """Test 1D UNet forward pass."""
        batch_size = 2
        in_channels = 1
        length = 128

        config = UNetConfig(
            architecture="UNet",
            conv_dims=1,
            in_channels=in_channels,
            num_classes=1,
            depth=3,
            num_channels_init=32,
        )
        model = UNet(**config.model_dump())

        # Create input (B, C, L)
        x = torch.randn(batch_size, in_channels, length)

        # Forward pass
        output = model(x)

        # Check output shape
        assert output.shape == (batch_size, 1, length)

    @pytest.mark.parametrize("in_channels", [1, 3, 4])
    @pytest.mark.parametrize("num_classes", [1, 2, 3])
    def test_unet1d_multichannel(self, in_channels: int, num_classes: int):
        """Test 1D UNet with different channel configurations."""
        batch_size = 2
        length = 64

        config = UNetConfig(
            architecture="UNet",
            conv_dims=1,
            in_channels=in_channels,
            num_classes=num_classes,
            depth=2,
            num_channels_init=32,
        )
        model = UNet(**config.model_dump())

        x = torch.randn(batch_size, in_channels, length)
        output = model(x)

        assert output.shape == (batch_size, num_classes, length)

    def test_unet1d_n2v2_mode(self):
        """Test 1D UNet with N2V2 mode (blur pool layers)."""
        config = UNetConfig(
            architecture="UNet",
            conv_dims=1,
            in_channels=1,
            num_classes=1,
            depth=2,
            num_channels_init=32,
            n2v2=True,
        )
        model = UNet(**config.model_dump())

        x = torch.randn(2, 1, 128)
        output = model(x)

        assert output.shape == x.shape

    def test_unet1d_independent_channels(self):
        """Test 1D UNet with independent channel processing."""
        config = UNetConfig(
            architecture="UNet",
            conv_dims=1,
            in_channels=3,
            num_classes=3,
            depth=2,
            num_channels_init=32,
            independent_channels=True,
        )
        model = UNet(**config.model_dump())

        x = torch.randn(2, 3, 64)
        output = model(x)

        assert output.shape == x.shape

    def test_unet1d_batch_norm(self):
        """Test 1D UNet with batch normalization."""
        config = UNetConfig(
            architecture="UNet",
            conv_dims=1,
            in_channels=1,
            num_classes=1,
            depth=2,
            num_channels_init=32,
            use_batch_norm=True,
        )
        model = UNet(**config.model_dump())

        x = torch.randn(2, 1, 64)
        output = model(x)

        assert output.shape == x.shape

    @pytest.mark.parametrize("length", [32, 64, 128, 256])
    def test_unet1d_variable_lengths(self, length: int):
        """Test 1D UNet handles different input lengths."""
        config = UNetConfig(
            architecture="UNet",
            conv_dims=1,
            in_channels=1,
            num_classes=1,
            depth=3,
            num_channels_init=32,
        )
        model = UNet(**config.model_dump())

        x = torch.randn(2, 1, length)
        output = model(x)

        assert output.shape == (2, 1, length)

    def test_unet1d_dtype_preservation(self):
        """Test 1D UNet preserves dtype."""
        config = UNetConfig(
            architecture="UNet",
            conv_dims=1,
            in_channels=1,
            num_classes=1,
            depth=2,
            num_channels_init=32,
        )
        model = UNet(**config.model_dump())

        for dtype in [torch.float32, torch.float64]:
            x = torch.randn(2, 1, 64, dtype=dtype)
            output = model(x)
            assert output.dtype == dtype

    def test_unet1d_gradient_flow(self):
        """Test gradients flow through 1D UNet."""
        config = UNetConfig(
            architecture="UNet",
            conv_dims=1,
            in_channels=1,
            num_classes=1,
            depth=2,
            num_channels_init=32,
        )
        model = UNet(**config.model_dump())

        x = torch.randn(2, 1, 64, requires_grad=True)
        output = model(x)

        # Compute loss and backward
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_unet1d_device_compatibility(self):
        """Test 1D UNet works on different devices."""
        config = UNetConfig(
            architecture="UNet",
            conv_dims=1,
            in_channels=1,
            num_classes=1,
            depth=2,
            num_channels_init=32,
        )
        model = UNet(**config.model_dump())

        # CPU
        x_cpu = torch.randn(2, 1, 64)
        output_cpu = model(x_cpu)
        assert output_cpu.device.type == "cpu"

        # GPU if available
        if torch.cuda.is_available():
            model_gpu = model.cuda()
            x_gpu = x_cpu.cuda()
            output_gpu = model_gpu(x_gpu)
            assert output_gpu.device.type == "cuda"

    def test_unet1d_mismatching_channels(self):
        """Test 1D UNet with mismatching in/out channels (for pixel embedding)."""
        config = UNetConfig(
            architecture="UNet",
            conv_dims=1,
            in_channels=1,
            num_classes=2,  # Output has more channels (pixel embedding)
            depth=2,
            num_channels_init=32,
        )
        model = UNet(**config.model_dump())

        x = torch.randn(2, 1, 64)
        output = model(x)

        assert output.shape == (2, 2, 64)
