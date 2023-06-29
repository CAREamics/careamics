import numpy as np
import pytest
import tifffile

from careamics_restoration.dataset.dataset import (
    PatchDataset,
)
from careamics_restoration.dataset.dataset_utils import (
    extract_patches_sequential,
    list_input_source_tiff,
)


def test_list_input_source_tiff(tmp_path):
    """Test listing tiff files"""
    num_files = 4

    # create np arrays
    arrays = []
    for n in range(num_files):
        arr_list = [list(range(2 + n)), [i * i for i in range(2 + n)]]
        arrays.append(np.array(arr_list))

    # create tiff files
    for n, arr in enumerate(arrays):
        tifffile.imwrite(tmp_path / f"test_{n}.tif", arr)

    # open tiff files without stating the number of files
    list_of_files = list_input_source_tiff(tmp_path)
    assert len(list_of_files) == num_files
    assert len(set(list_of_files)) == len(list_of_files)

    # open only 2 tiffs
    list_of_files = list_input_source_tiff(tmp_path, num_files=2)
    assert len(list_of_files) == 2
    assert len(set(list_of_files)) == 2


@pytest.mark.parametrize(
    "arr_shape, axes",
    [
        # wrong shapes
        ((8,), "Y"),
        # axes and array not compatible
        ((8, 8), "Y"),
        ((8, 8, 8), "YX"),
        ((8, 8, 8, 8, 8), "TCZYX"),
        # wrong axes
        ((8, 8), "XY"),
        ((8, 8, 8), "YXZ"),
        ((3, 8, 8), "CYX"),
        ((5, 3, 8, 8), "STYX"),
    ],
)
def test_patch_dataset_read_source_errors(tmp_path, arr_shape, axes):
    arr = np.ones(arr_shape)

    path = tmp_path / "test.tif"
    tifffile.imwrite(path, arr)
    assert path.exists()

    with pytest.raises((ValueError, NotImplementedError)):
        PatchDataset.read_tiff_source(path, axes)


@pytest.mark.parametrize(
    "arr_shape, axes",
    [
        # All possible input variations (S(B), T(o), C(o), Z(o), Y, X)
        # -> ((S(B) * T * P, C(o), Z(o), Y, X)
        # 2D (S(B), C, Y, X)
        # 3D (S(B), C, Z, Y, X)
        # 2D
        ((8, 8), "YX"),
        # ((2, 8, 8), "CYX", (4, 4)),
        # # 2D time series
        ((10, 8, 8), "TYX"),
        # (10, 1, 8, 8),
        # (10, 2, 8, 8),
        # # 3D
        ((4, 8, 8), "ZYX"),
        ((8, 8, 8), "ZYX"),
        # # 3D time series
        # (10, 32, 64, 64),
    ],
)
def test_patch_dataset_read_source(tmp_path, ordered_array, arr_shape, axes):
    arr = ordered_array(arr_shape)

    path = tmp_path / "test.tif"
    tifffile.imwrite(path, arr)
    assert path.exists()

    image = PatchDataset.read_tiff_source(path, axes)

    if axes == "YX":
        assert image.shape == (1, *arr_shape)
    elif axes == "CYX":
        assert image.shape == arr_shape[1:]
    elif axes == "TYX":
        assert image.shape == arr_shape


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
def test_extract_patches_sequential_invalid_arguments(arr_shape, patch_size):
    arr = np.zeros(arr_shape)

    with pytest.raises(ValueError):
        patches_generator = extract_patches_sequential(arr, patch_size)

        # get next yielded value
        next(patches_generator)


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

    The 3D array is a fixture of shape (1, 5, 10, 9)."""
    check_extract_patches_sequential(array_3D, patch_size)
