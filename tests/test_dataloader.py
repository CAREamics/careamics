import pytest
import tifffile

import numpy as np

from n2v.dataloader import (
    list_input_source_tiff,
    extract_patches_sequential,
    PatchDataset,
)


def test_list_input_source_tiff(tmp_path):
    """Test listing tiff files"""
    num_files = 4

    # create np arrays
    arrays = []
    for n in range(num_files):
        arr_list = [[i for i in range(2 + n)], [i * i for i in range(2 + n)]]
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
    "arr_shape, axes, patch_size",
    [
        # wrong shapes
        ((8,), "Y", (4,)),
        # axes and array not compatible
        ((8, 8), "Y", (4, 4)),
        ((8, 8, 8), "YX", (4, 4)),
        ((8, 8, 8, 8, 8), "TCZYX", (4, 4, 4)),
        # wrong axes
        ((8, 8), "XY", (4, 4)),
        ((8, 8, 8), "YXZ", (4, 4, 4)),
        ((3, 8, 8), "CYX", (1, 4, 4)),
        ((5, 3, 8, 8), "STYX", (1, 1, 4, 4)),
        # wrong patch size
        ((8, 8), "YX", (4,)),
        ((8, 8), "YX", (4, 3)),
        ((8, 8), "YX", (4, 4, 4)),
        ((8, 8, 8), "ZYX", (4, 4)),
        ((8, 8, 8), "ZYX", (4, 4, 4, 4)),
        ((8, 8, 8), "ZYX", (3, 4, 4)),
    ],
)
def test_patch_dataset_read_source_errors(tmp_path, arr_shape, axes, patch_size):
    arr = np.ones(arr_shape)

    path = tmp_path / "test.tif"
    tifffile.imwrite(path, arr)
    assert path.exists()

    with pytest.raises((ValueError, NotImplementedError)):
        PatchDataset.read_data_source(path, axes, patch_size)


@pytest.mark.parametrize(
    "arr_shape, axes, patch_size",
    [
        # All possible input variations (S(B), T(o), C(o), Z(o), Y, X)
        # -> ((S(B) * T * P, C(o), Z(o), Y, X)
        # 2D (S(B), C, Y, X)
        # 3D (S(B), C, Z, Y, X)
        # 2D
        ((8, 8), "YX", (4, 4)),
        # ((2, 8, 8), "CYX", (4, 4)),
        # # 2D time series
        ((10, 8, 8), "TYX", (4, 4)),
        # (10, 1, 8, 8),
        # (10, 2, 8, 8),
        # # 3D
        ((4, 8, 8), "ZYX", (4, 4, 4)),
        ((8, 8, 8), "ZYX", (4, 4, 4)),
        # # 3D time series
        # (10, 32, 64, 64),
    ],
)
def test_patch_dataset_read_source(
    tmp_path, ordered_array, arr_shape, axes, patch_size
):
    arr = ordered_array(arr_shape)

    path = tmp_path / "test.tif"
    tifffile.imwrite(path, arr)
    assert path.exists()

    image, updated_patch_size = PatchDataset.read_data_source(path, axes, patch_size)

    if axes == "YX":
        assert image.shape == (1,) + arr_shape
        assert updated_patch_size == (1,) + patch_size
    elif axes == "CYX":
        assert image.shape == arr_shape[1:]
        assert updated_patch_size == (1,) + patch_size
    elif axes == "TYX":
        assert image.shape == arr_shape
        assert updated_patch_size == (1,) + patch_size


@pytest.mark.parametrize(
    "arr_shape, patch_size",
    [
        # Wrong number of dimensions 2D
        ((10, 10), (5,)),
        ((10, 10), (5, 5)),  # minimum 3 dimensions CYX
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
    """Test extracting patches sequentially in 2D.

    The 2D array is a fixture of shape (1, 10, 9)."""
    patch_generator = extract_patches_sequential(array_2D, patch_size)

    # check patch shape
    patches = []
    for patch in patch_generator:
        patches.append(patch)
        assert patch.shape == (array_2D.shape[0],) + patch_size

    # we assume that if values are missing, these will be border ones
    # test can be simplified by checking all values, but then it might
    # get expensive to compute.
    patches_np = np.array(patches)
    for i in range(array_2D.shape[1]):
        assert array_2D[0, i, 0] in patches_np
        assert array_2D[0, i, -1] in patches_np

    for i in range(array_2D.shape[2]):
        assert array_2D[0, 0, i] in patches_np
        assert array_2D[0, -1, i] in patches_np


# TODO case (2, 3, 5), None doesn't work
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
    # compute expected number of patches
    patch_generator = extract_patches_sequential(array_3D, patch_size)

    # check individual patch shape
    patches = []
    for patch in patch_generator:
        patches.append(patch)
        assert patch.shape == (array_3D.shape[0],) + patch_size

    # we assume that if values are missing, these will be border ones
    # test can be simplified by checking all values, but then it might
    # get expensive to compute.
    shape_3d = array_3D.shape
    patches_np = np.array(patches)
    for i in range(shape_3d[2]):  # front and back of 3D cuboid
        for j in range(shape_3d[3]):
            assert array_3D[0, 0, i, j] in patches_np
            assert array_3D[0, -1, i, j] in patches_np

    for i in range(shape_3d[2]):  # top and bottom of 3D cuboid
        for k in range(shape_3d[1]):
            assert array_3D[0, k, i, 0] in patches_np
            assert array_3D[0, k, i, -1] in patches_np

    for j in range(shape_3d[3]):  # left and right of 3D cuboid
        for k in range(shape_3d[1]):
            assert array_3D[0, k, 0, j] in patches_np
            assert array_3D[0, k, -1, j] in patches_np
