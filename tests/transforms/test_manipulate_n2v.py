import numpy as np
import pytest
import torch

from careamics.config.support import SupportedPixelManipulation
from careamics.config.transformations import N2VManipulateConfig
from careamics.transforms import N2VManipulate, N2VManipulateTorch


@pytest.mark.parametrize(
    "strategy",
    [SupportedPixelManipulation.UNIFORM.value, SupportedPixelManipulation.MEDIAN.value],
)
def test_manipulate_n2v(strategy):
    """Test the N2V augmentation."""
    # create array, adding a channel to simulate a 2D image with channel last
    array = np.arange(16 * 16).reshape((16, 16))[np.newaxis, ...]

    # create augmentation
    aug = N2VManipulate(roi_size=5, masked_pixel_percentage=5, strategy=strategy)

    # apply augmentation
    augmented = aug(array)
    assert len(augmented) == 3  # transformed_patch, original_patch, mask

    # assert that the difference between the original and transformed patch are the
    # same pixels that are selected by the mask
    tr_path, orig_patch, mask = augmented
    diff_coords = np.array(np.where(tr_path != orig_patch))
    mask_coords = np.array(np.where(mask == 1))
    assert np.array_equal(diff_coords, mask_coords)


@pytest.mark.parametrize(
    "strategy",
    [SupportedPixelManipulation.UNIFORM, SupportedPixelManipulation.MEDIAN],
)
def test_manipulate_n2v_torch(strategy):
    """Test the N2V augmentation."""
    # Create tensor, adding a channel to simulate a 2D image with channel first
    array = torch.arange(16 * 16).reshape(1, 16, 16).float()

    # create configuration
    config = N2VManipulateConfig(
        roi_size=5, masked_pixel_percentage=5, strategy=strategy.value
    )
    # Create augmentation
    aug = N2VManipulateTorch(config, device="cpu")

    # Apply augmentation
    augmented = aug(array)
    assert len(augmented) == 3  # transformed_patch, original_patch, mask

    # Assert that the difference between the original and transformed patch are the
    # same pixels that are selected by the mask
    tr_patch, orig_patch, mask = augmented
    diff_coords = torch.nonzero(tr_patch != orig_patch, as_tuple=False).T
    mask_coords = torch.nonzero(mask == 1, as_tuple=False).T
    assert torch.equal(diff_coords, mask_coords)


def test_manipulate_n2v_torch_with_seed():
    """Test that seed is properly used for reproducibility."""
    # Create tensor, adding a channel to simulate a 2D image with channel first
    array = torch.arange(16 * 16).reshape(1, 16, 16).float()

    # Test with seed - should produce same masks
    config = N2VManipulateConfig(roi_size=5, masked_pixel_percentage=5, seed=1)
    aug1 = N2VManipulateTorch(config, device="cpu")
    aug2 = N2VManipulateTorch(config, device="cpu")

    _, _, mask1 = aug1(array)
    _, _, mask2 = aug2(array)

    # Same seed should produce same masks
    assert torch.equal(mask1, mask2)

    # Test without seed - should produce different masks
    config_no_seed = N2VManipulateConfig(
        roi_size=5, masked_pixel_percentage=5, seed=None
    )
    aug3 = N2VManipulateTorch(config_no_seed, device="cpu")

    _, _, mask3 = aug3(array)

    # No seed should produce different masks (very unlikely to be the same)
    assert not torch.equal(mask2, mask3)
