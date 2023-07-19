import numpy as np
import pytest

from careamics_restoration.dataset.tiling import (
    extract_tiles_predict,
    extract_patches_random,
    extract_patches_sequential,
)


@pytest.mark.parametrize(
    "arr_shape, patch_size",
    [
        # Wrong number of dimensions 2D
        ((10, 10), (5,)),
        ((10, 10), (5, 5)),
        # minimum 3 dimensions CYX
        ((10, 10), (5, 5, 5)),
        ((1, 10, 10), (5,)),
        ((1, 1, 10, 10), (5,)),
        # Wrong number of dimensions 3D
        ((10, 10, 10), (5, 5, 5, 5)),
        ((1, 10, 10, 10), (5, 5)),
        ((1, 10, 10, 10), (5, 5, 5, 5)),
        ((1, 1, 10, 10, 10), (5, 5)),
        ((1, 1, 10, 10, 10), (5, 5, 5, 5)),
        # Wrong z patch size
        ((1, 10, 10), (5, 5, 5)),
        ((10, 10, 10), (10, 5, 5)),
        # Wrong YX patch sizes
        ((1, 10, 10), (12, 5)),
        ((1, 10, 10), (5, 11)),
    ],
)
def test_extract_patches_invalid_arguments(arr_shape, patch_size):
    arr = np.zeros(arr_shape)

    with pytest.raises(ValueError):
        patches_generator = extract_patches_sequential(arr, patch_size)

        # get next yielded value
        next(patches_generator)

    with pytest.raises(ValueError):
        patches_generator = extract_patches_random(arr, patch_size)

        # get next yielded value
        next(patches_generator)


def test_extract_tiles_predict():
    """Test extracting patches randomly."""
    extract_tiles_predict()
    pass

@pytest.mark.parametrize(
    "arr_shape, patch_size",
    [
        # wrong number of dimensions
        ((8, 8), (2, 2)),
        ((1, 1, 8, 8, 8), (4, 4, 4)),
        # incompatible array and patch sizes
        ((1, 8, 8), (2,)),
        ((1, 8, 8), (2, 2, 2)),
        ((1, 8, 8, 8), (2, 2)),
        # patches with non power of two values
        ((1, 8, 8), (2, 3)),
        ((1, 8, 8), (5, 2)),
        ((1, 8, 8, 8), (5, 2, 2)),
        # patches with size 1
        ((1, 8, 8), (2, 1)),
        ((1, 8, 8), (1, 2)),
        ((1, 8, 8, 8), (1, 2, 2)),
        # patches too large
        ((1, 8, 8), (9, 9)),
        ((1, 8, 8, 8), (9, 9, 9)),
    ],
)
def test_extract_patches_sequential_errors(arr_shape, patch_size):
    """Test errors when trying to extract patches serquentially."""
    arr = np.zeros(arr_shape)

    with pytest.raises(ValueError):
        patches_generator = extract_patches_sequential(arr, patch_size)

        # get next yielded value
        next(patches_generator)


def check_extract_patches_sequential(array, patch_size):
    """Check that the patches are extracted correctly.

    The array should have been generated using np.arange and np.reshape."""
    patch_generator = extract_patches_sequential(array, patch_size)

    # check patch shape
    patches = []
    for patch in patch_generator:
        patches.append(patch)
        assert patch.shape == patch_size

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


def test_calculate_stats():
    arr = np.random.rand(2, 10, 10)

    mean = 0
    std = 0
    for i in range(arr.shape[0]):
        mean += np.mean(arr[i])
        std += np.std(arr[i])

    assert np.around(arr.mean(), decimals=4) == np.around(mean / (i + 1), decimals=4)
    assert np.around(arr.std(), decimals=2) == np.around(std / (i + 1), decimals=2)


# def test_extract_patches_random():
#     extract_patches_random()
#     pass

# def test_extract_patches_predict():
#     extract_patches_predict()
#     pass
