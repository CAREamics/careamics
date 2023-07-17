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
