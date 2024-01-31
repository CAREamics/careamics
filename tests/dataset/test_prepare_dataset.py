import numpy as np
import pytest

from careamics.dataset.dataset_utils import list_files, read_tiff, reshape_data

# TODO read source in one place, inside or outside dataset
# update axes for masks ?


def test_read_tiff(example_data_path):
    read_tiff(example_data_path[0], axes="SYX")


def test_read_tiff_raises(example_data_path):
    with pytest.raises(ValueError):
        read_tiff(example_data_path[1])


@pytest.mark.parametrize(
    "shape, axes, final_shape, final_axes",
    [
        ((16, 8), "YX", (1, 1, 16, 8), "SCYX"),
        ((16, 8), "XY", (1, 1, 8, 16), "SCYX"),
        ((16, 3, 8), "XZY", (1, 1, 3, 8, 16), "SCZYX"),
        ((16, 3, 8), "ZXY", (1, 1, 16, 8, 3), "SCZYX"),
        ((16, 3, 12), "SXY", (16, 1, 12, 3), "SCYX"),
        ((5, 5, 2), "XYS", (2, 1, 5, 5), "SCYX"),
        ((5, 1, 5, 2), "XZYS", (2, 1, 1, 5, 5), "SCZYX"),
        ((5, 12, 5, 2), "ZXYS", (2, 1, 5, 5, 12), "SCZYX"),
        ((16, 8, 5, 12), "SZYX", (16, 1, 8, 5, 12), "SCZYX"),
        ((16, 8, 5), "YXT", (5, 1, 16, 8), "SCYX"),  # T, no C
        ((4, 16, 8), "TXY", (4, 1, 8, 16), "SCYX"),
        ((4, 16, 6, 8), "TXSY", (4 * 6, 1, 8, 16), "SCYX"),
        ((4, 16, 6, 5, 8), "ZXTYS", (8 * 6, 1, 4, 5, 16), "SCZYX"),
        ((5, 3, 5), "XCY", (1, 3, 5, 5), "SCYX"),  # C, no T
        ((16, 3, 12, 8), "XCYS", (8, 3, 12, 16), "SCYX"),
        ((16, 12, 3, 8), "ZXCY", (1, 3, 16, 8, 12), "SCZYX"),
        ((16, 3, 12, 8), "XCYZ", (1, 3, 8, 12, 16), "SCZYX"),
        ((16, 8, 12, 3), "ZYXC", (1, 3, 16, 8, 12), "SCZYX"),
        ((16, 8, 21, 12, 3), "ZYSXC", (21, 3, 16, 8, 12), "SCZYX"),
        ((16, 21, 8, 3, 12), "SZYCX", (16, 3, 21, 8, 12), "SCZYX"),
        ((5, 3, 8, 6), "XTCY", (3, 8, 6, 5), "SCYX"),  # CT
        ((16, 3, 12, 5, 8), "XCYTS", (8 * 5, 3, 12, 16), "SCYX"),
        ((16, 10, 5, 6, 12, 8), "ZSXCYT", (10 * 8, 6, 16, 12, 5), "SCZYX"),
    ],
)
def test_update_axes(shape, axes, final_shape, final_axes):
    array = np.zeros(shape)

    new_array, new_axes = reshape_data(array, axes)
    assert new_array.shape == final_shape
    assert new_axes == final_axes


@pytest.mark.parametrize(
    "shape, axes",
    [
        ((1, 16, 8), "YX"),
        ((1, 16, 3, 8), "XZY"),
    ],
)
def test_update_axes_raises(shape, axes):
    array = np.zeros(shape)

    with pytest.raises(ValueError):
        reshape_data(array, axes)


def test_list_files(example_data_path):
    train_path, _, _ = example_data_path

    files_from_path = list_files(train_path, "tif")
    assert len(files_from_path) >= 1
    assert all(file.suffix == ".tif" for file in files_from_path)

    files_from_str = list_files(train_path._str, "tif")
    assert len(files_from_str) >= 1
    assert all(file.suffix == ".tif" for file in files_from_str)

    files_from_list = list_files([train_path], "tif")
    assert len(files_from_list) >= 1
    assert all(file.suffix == ".tif" for file in files_from_list)


def test_list_files_raises(example_data_path):
    train_path, _, _ = example_data_path

    with pytest.raises(ValueError):
        list_files(train_path, "jpg")
        list_files(train_path.name, "tif")
