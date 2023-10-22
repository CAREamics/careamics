import pytest

from careamics.dataset.dataset_utils import (
    list_files,
)


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
