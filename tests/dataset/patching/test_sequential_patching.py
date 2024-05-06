import numpy as np
import pytest

from careamics.dataset.patching.sequential_patching import (
    _compute_number_of_patches,
    _compute_overlap,
    _compute_patch_steps,
    _compute_patch_views,
    extract_patches_sequential,
)


def check_extract_patches_sequential(array: np.ndarray, patch_size: tuple):
    """Check that the patches are extracted correctly.

    The array should have been generated using np.arange and np.reshape."""
    patches, _ = extract_patches_sequential(array, patch_size=patch_size)

    # check patch shape
    assert patches.shape[2:] == patch_size

    # check that all values are covered by the patches
    n_max = np.prod(array.shape)  # maximum value in the array
    unique = np.unique(np.array(patches))  # unique values in the patches
    assert len(unique) == n_max


@pytest.mark.parametrize(
    "patch_size",
    [
        (2, 2),
        (4, 2),
        (4, 8),
        (8, 8),
    ],
)
def test_extract_patches_sequential_2d(array_2D, patch_size):
    """Test extracting patches sequentially in 2D."""
    check_extract_patches_sequential(array_2D, patch_size)


@pytest.mark.parametrize(
    "patch_size",
    [
        (2, 2, 4),
        (4, 2, 2),
        (2, 8, 4),
        (4, 8, 8),
    ],
)
def test_extract_patches_sequential_3d(array_3D, patch_size):
    """Test extracting patches sequentially in 3D.

    The 3D array is a fixture of shape (1, 8, 16, 16)."""
    # TODO changed the fixture to (1, 8, 16, 16), uneven shape doesnt work. We need to
    # discuss the function or the test cases
    check_extract_patches_sequential(array_3D, patch_size)


@pytest.mark.parametrize(
    "patch_size",
    [
        (4, 4),
        (8, 8),
    ],
)
def test_extract_patches_sequential_2d_supervised(array_2D, patch_size):
    """Test extracting patches sequentially in 2D with target."""
    patches, targets = extract_patches_sequential(
        array_2D, patch_size=patch_size, target=array_2D
    )

    # Check that the same region is extracted in the patches and targets
    assert np.array_equal(patches, targets)


@pytest.mark.parametrize(
    "shape, patch_sizes, expected",
    [
        ((1, 3, 10, 10), (1, 3, 10, 5), (1, 1, 1, 2)),
        ((1, 1, 9, 9), (1, 1, 4, 3), (1, 1, 3, 3)),
        ((1, 3, 10, 9), (1, 3, 3, 5), (1, 1, 4, 2)),
        ((1, 1, 5, 9, 10), (1, 1, 2, 3, 5), (1, 1, 3, 3, 2)),
    ],
)
def test_compute_number_of_patches(shape, patch_sizes, expected):
    """Test computing number of patches"""
    assert _compute_number_of_patches(shape, patch_sizes) == expected


@pytest.mark.parametrize(
    "shape, patch_sizes, expected",
    [
        ((1, 3, 10, 10), (1, 3, 10, 5), (0, 0, 0, 0)),
        ((1, 1, 9, 9), (1, 1, 4, 3), (0, 0, 2, 0)),
        ((1, 3, 10, 10, 9), (1, 3, 2, 3, 5), (0, 0, 0, 1, 1)),
    ],
)
def test_compute_overlap(shape, patch_sizes, expected):
    """Test computing overlap between patches"""
    assert _compute_overlap(shape, patch_sizes) == expected


@pytest.mark.parametrize("dims", [2, 3])
@pytest.mark.parametrize("patch_size", [2, 3])
@pytest.mark.parametrize("overlap", [0, 1, 4])
def test_compute_patch_steps(dims, patch_size, overlap):
    """Test computing patch steps"""
    patch_sizes = (patch_size,) * dims
    overlaps = (overlap,) * dims
    expected = (min(patch_size - overlap, patch_size),) * dims

    assert _compute_patch_steps(patch_sizes, overlaps) == expected


def check_compute_reshaped_view(array: np.ndarray, window_shape, steps):
    """Check the number of patches"""

    output_shape = (-1, *window_shape)

    # compute views
    output = _compute_patch_views(array, window_shape, steps, output_shape)

    # check the number of patches
    n_patches = [
        np.ceil((array.shape[i] - window_shape[i] + 1) / steps[i]).astype(int)
        for i in range(len(window_shape))
    ]
    assert output.shape == (np.prod(n_patches), *window_shape)


@pytest.mark.parametrize(
    "window_shape, steps",
    [
        ((1, 3, 5, 5), (1, 1, 1, 1)),
        ((1, 3, 5, 5), (1, 1, 2, 3)),
        ((1, 3, 5, 7), (1, 1, 1, 1)),
    ],
)
def test_compute_reshaped_view_2d(array_2D, window_shape, steps):
    """Test computing reshaped view of an array of shape (1, 10, 9)."""
    check_compute_reshaped_view(array_2D, window_shape, steps)


@pytest.mark.parametrize(
    "window_shape, steps",
    [
        ((1, 3, 1, 5, 5), (1, 1, 2, 1, 2)),
        ((1, 3, 2, 5, 5), (1, 1, 2, 3, 4)),
        ((1, 3, 3, 7, 8), (1, 1, 1, 1, 3)),
    ],
)
def test_compute_reshaped_view_3d(array_3D, window_shape, steps):
    """Test computing reshaped view of an array of shape (1, 5, 10, 9)."""
    check_compute_reshaped_view(array_3D, window_shape, steps)
