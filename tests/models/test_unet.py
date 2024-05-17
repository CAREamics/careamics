import pytest
import torch

from careamics.models.layers import MaxBlurPool
from careamics.models.unet import UNet


@pytest.mark.parametrize("depth", [1, 3, 5])
def test_unet_depth(depth):
    """Test that the UNet has the correct number of down and up convs
    with respect to the depth."""
    model = UNet(conv_dims=2, depth=depth)

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


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 1, 1024, 1024),
        (1, 1, 512, 512),
        (1, 1, 256, 256),
        (1, 1, 128, 128),
        (1, 1, 64, 64),
        (1, 1, 32, 32),
        (1, 1, 16, 16),
        (1, 1, 8, 8),
    ],
)
def test_blurpool2d(input_shape):
    """Test that the BlurPool2d layer works as expected."""
    layer = MaxBlurPool(dim=2, kernel_size=3)
    assert layer(torch.randn(input_shape)).shape == tuple(
        [1, 1] + [i // 2 for i in input_shape[2:]]
    )


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 1, 256, 256, 256),
        (1, 1, 128, 128, 128),
        (1, 1, 64, 128, 128),
        (1, 1, 64, 64, 64),
        (1, 1, 32, 64, 64),
        (1, 1, 32, 32, 32),
        (1, 1, 16, 16, 16),
        (1, 1, 8, 8, 8),
    ],
)
def test_blurpool3d(input_shape):
    """Test that the BlurPool3d layer works as expected."""
    layer = MaxBlurPool(dim=3, kernel_size=3)
    assert layer(torch.randn(input_shape)).shape == tuple(
        [1, 1] + [i // 2 for i in input_shape[2:]]
    )


@pytest.mark.parametrize("independent_channels", [True, False])
def test_independent_channels(independent_channels):
    """
    Test that with independent channels there is no connections between the
    parallel networks.
    """

    n_channels = 2
    input_shape = (1, n_channels, 8, 8)

    # input 1
    x1 = torch.randn(input_shape)
    # input 2: channel 1 same as input 1, channel 2 all zeros
    x2 = x1.clone()
    x2[:, 1] = 0

    model = UNet(
        conv_dims=2,
        in_channels=n_channels,
        num_classes=n_channels,
        independent_channels=independent_channels,
    )
    model.eval()

    y1 = model(x1)
    y2 = model(x2)

    # channel 2 different between inputs => channel 2 different between outputs
    assert (y1[:, 1] != y2[:, 1]).any()

    if independent_channels:
        # when channel independent
        # channel 1 same between inputs => channel 1 same between outputs
        assert (y1[:, 0] == y2[:, 0]).all()
    else:
        # when not independent
        # channel 2 different between inputs => all channels different between outputs
        assert (y1[:, 0] != y2[:, 0]).any()
