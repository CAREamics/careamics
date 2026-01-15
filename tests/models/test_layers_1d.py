"""Tests for 1D layers."""

import pytest
import torch

from careamics.models.layers import MaxBlurPool


class TestMaxBlurPool1D:
    """Test MaxBlurPool layer with 1D data."""

    @pytest.mark.parametrize("kernel_size", [3, 5])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("max_pool_size", [2, 3])
    def test_maxblurpool1d_output_shape(
        self, kernel_size: int, stride: int, max_pool_size: int
    ):
        """Test MaxBlurPool1D produces correct output shape."""
        batch_size = 4
        channels = 3
        length = 64

        # Create 1D input (B, C, L)
        x = torch.randn(batch_size, channels, length)

        # Create MaxBlurPool with dim=1 for 1D data
        layer = MaxBlurPool(
            dim=1, kernel_size=kernel_size, stride=stride, max_pool_size=max_pool_size
        )

        # Forward pass
        output = layer(x)

        # Check output shape
        assert output.ndim == 3
        assert output.shape[0] == batch_size
        assert output.shape[1] == channels
        # Output length depends on max_pool_size and stride
        assert output.shape[2] <= length

    def test_maxblurpool1d_forward_pass(self):
        """Test MaxBlurPool1D forward pass runs without error."""
        x = torch.randn(2, 4, 128)
        layer = MaxBlurPool(dim=1, kernel_size=3, stride=2, max_pool_size=2)
        output = layer(x)

        # Should downsample
        assert output.shape[2] < x.shape[2]
        # Should preserve batch and channels
        assert output.shape[0] == x.shape[0]
        assert output.shape[1] == x.shape[1]

    def test_maxblurpool1d_dtype_preservation(self):
        """Test MaxBlurPool1D preserves dtype."""
        for dtype in [torch.float32, torch.float64]:
            x = torch.randn(2, 3, 64, dtype=dtype)
            layer = MaxBlurPool(dim=1, kernel_size=3, stride=2)
            output = layer(x)
            assert output.dtype == dtype

    def test_maxblurpool1d_device_compatibility(self):
        """Test MaxBlurPool1D works on different devices."""
        x = torch.randn(2, 3, 64)
        layer = MaxBlurPool(dim=1, kernel_size=3, stride=2)

        # CPU
        output_cpu = layer(x)
        assert output_cpu.device.type == "cpu"

        # GPU if available
        if torch.cuda.is_available():
            x_gpu = x.cuda()
            layer_gpu = layer.cuda()
            output_gpu = layer_gpu(x_gpu)
            assert output_gpu.device.type == "cuda"

    def test_maxblurpool1d_ceil_mode(self):
        """Test ceil_mode parameter affects output size."""
        x = torch.randn(2, 3, 65)  # Odd length
        layer_floor = MaxBlurPool(dim=1, kernel_size=3, stride=2, ceil_mode=False)
        layer_ceil = MaxBlurPool(dim=1, kernel_size=3, stride=2, ceil_mode=True)

        output_floor = layer_floor(x)
        output_ceil = layer_ceil(x)

        # Ceil mode may produce different output size
        assert output_ceil.shape[2] >= output_floor.shape[2]
