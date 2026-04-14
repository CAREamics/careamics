import pytest
import torch

from careamics.config.augmentations import N2VManipulateConfig
from careamics.config.support import SupportedPixelManipulation
from careamics.transforms import N2VManipulate


@pytest.mark.parametrize(
    "strategy",
    [SupportedPixelManipulation.UNIFORM, SupportedPixelManipulation.MEDIAN],
)
def test_manipulate_n2v(strategy):
    """Test the N2V augmentation."""
    # Create tensor, adding a channel to simulate a 2D image with channel first
    array = torch.arange(16 * 16).reshape(1, 16, 16).float()

    # create configuration
    config = N2VManipulateConfig(
        roi_size=5, masked_pixel_percentage=5, strategy=strategy.value
    )
    # Create augmentation
    aug = N2VManipulate(config, device="cpu")

    # Apply augmentation
    augmented = aug(array)
    assert len(augmented) == 3  # transformed_patch, original_patch, mask

    # Assert that the difference between the original and transformed patch are the
    # same pixels that are selected by the mask
    tr_patch, orig_patch, mask = augmented
    diff_coords = torch.nonzero(tr_patch != orig_patch, as_tuple=False).T
    mask_coords = torch.nonzero(mask == 1, as_tuple=False).T
    assert torch.equal(diff_coords, mask_coords)


def test_manipulate_n2v_with_seed():
    """Test that seed is properly used for reproducibility."""
    array = torch.arange(16 * 16).reshape(1, 16, 16).float()

    config = N2VManipulateConfig(roi_size=5, masked_pixel_percentage=5, seed=1)
    aug1 = N2VManipulate(config, device="cpu")
    aug2 = N2VManipulate(config, device="cpu")

    _, _, mask1 = aug1(array)
    _, _, mask2 = aug2(array)

    # Same seed should produce same masks
    assert torch.equal(mask1, mask2)

    # Subsequent calls produce a different mask
    _, _, mask3 = aug2(array)
    assert not torch.equal(mask2, mask3)
