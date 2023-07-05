import pytest

from careamics_restoration.models.unet import UNet

# TODO mode tests
# TODO: test n2v2 and skip_skipone
# TODO: test various parameters


@pytest.mark.parametrize("depth", [1, 3, 5])
def test_unet_depth(depth):
    """Test that the UNet has the correct number of down and up convs
    with respect to the depth."""
    model = UNet(conv_dim=2, depth=depth)

    # check depth
    assert model.depth == depth

    # check that encoder has the right number of down convs
    counter_pool = 0
    for layer in model.enc_blocks.keys():
        if "pool" in layer:
            counter_pool += 1

    assert counter_pool == depth

    # check that decoder has the right number of up convs
    counter_up = 0
    for layer in model.dec_blocks.keys():
        if "upsampling" in layer:
            counter_up += 1

    assert counter_up == depth


@pytest.mark.parametrize("depth", [0, -1])
def test_no_depth_error(depth):
    """Test that the UNet raises an error if the depth is less than 1."""
    with pytest.raises(ValueError):
        UNet(conv_dim=2, depth=depth)
