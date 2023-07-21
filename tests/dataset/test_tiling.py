import numpy as np
import pytest

from careamics_restoration.dataset.tiling import (
    extract_patches_random,
    extract_patches_sequential,
    extract_tiles,
    patches_sanity_check,
)


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


def check_extract_patches_random(array, patch_size):
    """Check that the patches are extracted correctly.

    The array should have been generated using np.arange and np.reshape."""
    patch_generator = extract_patches_random(array, patch_size)

    # check patch shape
    patches = []
    for patch in patch_generator:
        patches.append(patch)
        assert patch.shape == patch_size


def check_extract_tiles(array, tile_size, overlaps):
    """Test extracting patches randomly."""
    tile_data_generator = extract_tiles(array, tile_size, overlaps)

    tiles = []
    all_overlap_crop_coords = []
    all_stitch_coords = []
    # Assemble all tiles and their respective coordinates
    for tile_data in tile_data_generator:
        tile, _, _, overlap_crop_coords, stitch_coords = tile_data
        tiles.append(tile)
        all_overlap_crop_coords.append(overlap_crop_coords)
        all_stitch_coords.append(stitch_coords)

        # check tile shape, ignore sample dimension
        assert tile.shape[1:] == tile_size
        assert len(overlap_crop_coords) == len(stitch_coords) == len(tile_size)

    # check that each tile has a unique set of coordinates
    assert len(tiles) == len(all_overlap_crop_coords) == len(all_stitch_coords)

    # check that all values are covered by the tiles
    n_max = np.prod(array.shape)  # maximum value in the array
    unique = np.unique(np.array(tiles))  # unique values in the patches
    assert len(unique) >= n_max


@pytest.mark.parametrize(
    "arr_shape, patch_size",
    [
        ((1, 8, 8), (2, 2)),
        ((1, 8, 8, 8), (2, 2, 2)),
    ],
)
def test_patches_sanity_check(arr_shape, patch_size):
    arr = np.zeros(arr_shape)
    is_3d_patch = len(patch_size) == 3
    # check if the patch is 2D or 3D. Subtract 1 because the first dimension is sample
    patches_sanity_check(arr, patch_size, is_3d_patch)


@pytest.mark.parametrize(
    "arr_shape, patch_size",
    [
        # Wrong number of dimensions 2D
        ((10, 10), (5, 5)),
        # minimum 3 dimensions CYX
        ((10, 10), (5, 5, 5)),
        ((1, 1, 10, 10), (5, 5)),
        # Wrong number of dimensions 3D
        ((10, 10, 10), (5, 5, 5)),
        ((1, 10, 10, 10), (5, 5)),
        ((1, 1, 10, 10, 10), (5, 5)),
        ((1, 1, 10, 10, 10), (5, 5, 5)),
        # Wrong z patch size
        ((1, 10, 10), (5, 5, 5)),
        ((10, 10, 10), (10, 5, 5)),
        # Wrong YX patch sizes
        ((1, 10, 10), (12, 5)),
        ((1, 10, 10), (5, 11)),
    ],
)
def test_patches_sanity_check_invalid_cases(arr_shape, patch_size):
    arr = np.zeros(arr_shape)
    is_3d_patch = len(patch_size) == 3
    # check if the patch is 2D or 3D. Subtract 1 because the first dimension is sample
    with pytest.raises(ValueError):
        patches_sanity_check(arr, patch_size, is_3d_patch)


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
        (2, 2),
        (4, 2),
        (4, 8),
        (8, 8),
    ],
)
def test_extract_patches_random_2d(array_2D, patch_size):
    """Test extracting patches randomly in 2D."""
    check_extract_patches_random(array_2D, patch_size)


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
    check_extract_patches_random(array_3D, patch_size)


@pytest.mark.parametrize(
    "tile_size, overlaps",
    [
        ((4, 4), (2, 2)),
        ((8, 8), (4, 4)),
    ],
)
def test_extract_tiles_2d(array_2D, tile_size, overlaps):
    """Test extracting tiles for prediction in 2D."""
    check_extract_tiles(array_2D, tile_size, overlaps)


@pytest.mark.parametrize(
    "tile_size, overlaps",
    [
        ((4, 4, 4), (2, 2, 2)),
        ((8, 8, 8), (4, 4, 4)),
    ],
)
def test_extract_tiles_3d(array_3D, tile_size, overlaps):
    """Test extracting tiles for prediction in 3D.

    The 3D array is a fixture of shape (1, 8, 16, 16)."""
    check_extract_tiles(array_3D, tile_size, overlaps)


def test_calculate_stats():
    arr = np.random.rand(2, 10, 10)

    mean = 0
    std = 0
    for i in range(arr.shape[0]):
        mean += np.mean(arr[i])
        std += np.std(arr[i])

    assert np.around(arr.mean(), decimals=4) == np.around(mean / (i + 1), decimals=4)
    assert np.around(arr.std(), decimals=2) == np.around(std / (i + 1), decimals=2)
