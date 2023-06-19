import pytest
import torch.nn as nn

from careamics_restoration.models.models import UNET, UP_MODE, MERGE_MODE


@pytest.mark.parametrize("depth", [1, 3, 5])
def test_unet_depth(depth):
    """Test that the UNet has the correct number of down and up convs
    with respect to the depth."""
    model = UNET(conv_dim=2, depth=depth)

    assert len(model.down_convs) == depth
    assert len(model.up_convs) == depth - 1


@pytest.mark.parametrize("depth", [0, -1])
def test_no_depth_error(depth):
    """Test that the UNet raises an error if the depth is less than 1."""
    with pytest.raises(ValueError):
        UNET(conv_dim=2, depth=depth)


@pytest.mark.parametrize("up_mode", ["upsample", "transpose"])
def test_up_mode(up_mode):
    """Test that the UNet has the correct upsampling layers in the
    upsampling branch."""
    model = UNET(conv_dim=2, up_mode=up_mode, merge_mode=MERGE_MODE.CONCAT)

    # check that it recorded the correct value
    assert model.up_mode == up_mode

    # check the upsampling branch
    if up_mode == UP_MODE.TRANSPOSE:
        for up_conv in model.up_convs:
            assert isinstance(next(up_conv.children()), nn.ConvTranspose2d)
    elif up_mode == UP_MODE.UPSAMPLE:
        for up_conv in model.up_convs:
            assert isinstance(next(up_conv.children())[0], nn.Upsample)


@pytest.mark.parametrize("merge_mode", ["concat", "add"])
def test_merge_mode(merge_mode):
    """Test that the UNet has the correct merge layers in the upsampling
    branch."""
    model = UNET(conv_dim=2, up_mode=UP_MODE.TRANSPOSE, merge_mode=merge_mode)

    # check that it recorded the correct value
    assert model.merge_mode == merge_mode

    # compute channels in the upsampling
    for i in range(model.depth - 1):
        for up_conv in model.up_convs:
            iterator = up_conv.children()
            _ = next(iterator)
            middle_layer = next(iterator)

            # check that the middle convolution has twice the number of channels
            # in than out
            if merge_mode == MERGE_MODE.CONCAT:
                assert middle_layer.in_channels == 2 * middle_layer.out_channels
            # check that the middle convolution has the same number of channels
            # in and out
            elif merge_mode == MERGE_MODE.ADD:
                assert middle_layer.in_channels == middle_layer.out_channels


def test_upsample_add_error():
    """Test that the UNet raises an error if the up_mode is 'upsample' and
    the merge_mode is 'add'."""
    with pytest.raises(ValueError):
        UNET(conv_dim=2, up_mode=UP_MODE.UPSAMPLE, merge_mode=MERGE_MODE.ADD)
