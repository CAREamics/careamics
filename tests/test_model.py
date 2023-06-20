import pytest
import torch.nn as nn

from careamics_restoration.models.models import UNET


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
