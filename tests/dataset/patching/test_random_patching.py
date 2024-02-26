import pytest

import numpy as np
from careamics.dataset.patching.patching import extract_patches_random

# TODO this is just checking the size
def check_extract_patches_random(array: np.ndarray, axes, patch_size, target=None):
    """Check that the patches are extracted correctly.

    The array should have been generated using np.arange and np.reshape."""

    patch_generator = extract_patches_random(
        array, axes=axes, patch_size=patch_size, target=target
    )

    # check patch shape
    for patch, target in patch_generator:
        assert patch.shape[2:] == patch_size
        if target is not None:
            assert target.shape[2:] == patch_size



@pytest.mark.parametrize(
    "patch_size",
    [
        (2, 2),
        (4, 2),
        (4, 8),
        (8, 8),
    ],
)
def test_extract_patches_random_2d(array_2D, patch_size):
    """Test extracting patches randomly in 2D."""
    check_extract_patches_random(array_2D, "SYX", patch_size)


@pytest.mark.parametrize(
    "patch_size",
    [
        (2, 2),
        (4, 2),
        (4, 8),
        (8, 8),
    ],
)
def test_extract_patches_random_supervised_2d(array_2D, patch_size):
    """Test extracting patches randomly in 2D."""
    check_extract_patches_random(
        array_2D,
        "SYX",
        patch_size,
        target=array_2D
    )


@pytest.mark.parametrize(
    "patch_size",
    [
        (2, 2, 4),
        (4, 2, 2),
        (2, 8, 4),
        (4, 8, 8),
    ],
)
def test_extract_patches_random_3d(array_3D, patch_size):
    """Test extracting patches randomly in 3D.

    The 3D array is a fixture of shape (1, 8, 16, 16)."""
    check_extract_patches_random(array_3D, "SZYX", patch_size)
