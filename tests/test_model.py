import pytest
import torch

from careamics.models.layers import MaxBlurPool2D, MaxBlurPool3D
from careamics.models.unet import UNet


@pytest.mark.parametrize("depth", [1, 3, 5])
def test_unet_depth(depth):
    """Test that the UNet has the correct number of down and up convs
    with respect to the depth."""
    model = UNet(conv_dim=2, depth=depth)

    # check that encoder has the right number of down convs
    counter_down = 0
    for _, layer in model.encoder.encoder_blocks.named_children():
        if type(layer).__name__ == "Conv_Block":
            counter_down += 1

    assert counter_down == depth

    # check that decoder has the right number of up convs
    counter_up = 0
    for _, layer in model.decoder.decoder_blocks.named_children():
        if type(layer).__name__ == "Conv_Block":
            counter_up += 1

    assert counter_up == depth


def test_blurpool2d():
    """Test that the BlurPool2d layer works as expected."""
    layer = MaxBlurPool2D(kernel_size=3)
    assert layer(torch.randn(1, 1, 32, 32)).shape == (1, 1, 16, 16)


def test_blurpool3d():
    """Test that the BlurPool3d layer works as expected."""
    layer = MaxBlurPool3D(kernel_size=3)
    assert layer(torch.randn(1, 1, 32, 32, 32)).shape == (1, 1, 16, 16, 16)
