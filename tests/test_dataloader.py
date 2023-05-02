import pytest
import tifffile

import numpy as np

from n2v.dataloader import (
    list_input_source_tiff,
    extract_patches_sequential,
    PatchDataset,
)
from n2v.dataloader_utils.dataloader_utils import _compute_number_of_patches


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
        ((8, 8), "YX", (4, 4)),
    ],
)
def test_patch_dataset_read_source_errors(tmp_path, arr_shape, axes, patch_size):
    arr = np.arange(np.prod(arr_shape)).reshape(arr_shape)

    path = tmp_path / "test.tif"
    tifffile.imwrite(path, arr)
    assert path.exists()

    with pytest.raises(ValueError):
        PatchDataset.read_data_source(path, patch_size)


@pytest.mark.parametrize(
    "arr_shape, axes, patch_size",
    [
        # All possible input variations (S(B), T(o), C(o), Z(o), Y, X)
        # -> ((S(B) * T * P, C(o), Z(o), Y, X)
        # 2D (S(B), C, Y, X)
        # 3D (S(B), C, Z, Y, X)
        # 2D
        ((8, 8), "YX", (4, 4)),
        ((1, 8, 8), "CYX", (4, 4)),
        # ((2, 8, 8), "CYX", (4, 4)), This should fail
        # # 2D time series
        ((10, 8, 8), "TYX", (4, 4)),
        # (10, 1, 8, 8),
        # (10, 2, 8, 8),
        # # 3D
        ((4, 8, 8), "ZYX", (4, 4, 4)),
        ((8, 8, 8), "ZYX", (4, 4, 4)),
        ((1, 4, 8, 8), "CZYX", (4, 4, 4)),
        # # 3D time series
        # (10, 32, 64, 64),
    ],
)
def test_patch_dataset_read_source(tmp_path, arr_shape, axes, patch_size):
    arr = np.arange(np.prod(arr_shape)).reshape(arr_shape)

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


@pytest.mark.parametrize("overlaps", [(3, 2), (2, 1), None])
@pytest.mark.parametrize("patch_size", [(5, 5), (6, 3), (6, 6)])
def test_extract_patches_sequential_2d(array_2D, patch_size, overlaps):
    """Test extracting patches sequentially in 2D"""
    patch_generator = extract_patches_sequential(array_2D, patch_size, overlaps)

    # check patch shape
    counter = 0
    for patch in patch_generator:
        assert patch.shape == (array_2D.shape[0],) + patch_size

        counter += 1

    # check number of patches obtained
    if overlaps is None:
        n_patches = _compute_number_of_patches(array_2D, patch_size)
    else:
        n_patches = [
            (array_2D.shape[i + 1] - patch_size[i]) // (patch_size[i] - overlaps[i]) + 1
            for i in range(len(patch_size))
        ]

    assert counter == np.product(n_patches)


# TODO case (2, 3, 5), None doesn't work
@pytest.mark.parametrize("patch_size", [(3, 5, 5), (5, 5, 5), (3, 3, 5), (4, 6, 6)])
@pytest.mark.parametrize("overlaps", [(0, 2, 1), (1, 1, 2), (1, 2, 1), (2, 1, 2), None])
def test_extract_patches_sequential_3d(array_3D, patch_size, overlaps):
    """Test extracting patches sequentially in 3D"""
    # compute expected number of patches
    patch_generator = extract_patches_sequential(array_3D, patch_size, overlaps)

    # check individual patch shape
    counter = 0
    for patch in patch_generator:
        counter += 1
        assert patch.shape == (array_3D.shape[0],) + patch_size

    # check number of patches obtained
    if overlaps is None:
        n_patches = _compute_number_of_patches(array_3D, patch_size)
    else:
        n_patches = [
            (array_3D.shape[i + 1] - patch_size[i]) // (patch_size[i] - overlaps[i]) + 1
            for i in range(len(patch_size))
        ]

    assert counter == np.product(n_patches)
