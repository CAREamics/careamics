import pytest

import numpy as np
import tifffile

from careamics.config import DataModel
from careamics.config.support import SupportedData
from careamics.dataset import IterableDataset


def test_number_of_files(tmp_path, ordered_array):
    """Test number of files in IterableDataset."""
    # create array
    array = ordered_array((20, 20))
    
    # save three files
    file1 = tmp_path / "array1.tif"
    file2 = tmp_path / "array2.tif"
    file3 = tmp_path / "array3.tif"
    tifffile.imwrite(file1, array)
    tifffile.imwrite(file2, array)
    tifffile.imwrite(file3, array)

    # create config
    config_dict =  {
        "data_type": SupportedData.TIFF.value,
        "patch_size": [4, 4],
        "axes": "YX",
    }
    config = DataModel(**config_dict)

    # create dataset
    dataset = IterableDataset(
        data_config=config,
        src_files=[
            file1,
            file2,
            file3,
        ],
    )

    # check number of patches
    assert dataset.get_number_of_files() == 3
    assert dataset.get_number_of_files() == len(dataset.data_files)


@pytest.mark.parametrize("percentage", [0.1, 0.6])
def test_extracting_val_files(tmp_path, ordered_array, percentage):
    """Test extracting a validation set patches from InMemoryDataset."""
    # create array
    array = ordered_array((20, 20))

    # save array to 25 files
    files = []
    for i in range(25):
        file_path = tmp_path / f"array{i}.tif"
        tifffile.imwrite(file_path, array)
        files.append(file_path)

    # create config
    config_dict =  {
        "data_type": SupportedData.TIFF.value,
        "patch_size": [4, 4],
        "axes": "YX",
    }
    config = DataModel(**config_dict)

    # create dataset
    dataset = IterableDataset(
        data_config=config,
        src_files=files,
    )

    # compute number of patches
    total_n_files = dataset.get_number_of_files()
    minimum_files = 5
    n_files = max(round(percentage*total_n_files), minimum_files)

    # extract datset
    valset = dataset.split_dataset(percentage, minimum_files)

    # check number of patches
    assert valset.get_number_of_files() == n_files
    assert dataset.get_number_of_files() == total_n_files - n_files

    # check that none of the validation files are in the original dataset
    assert set(valset.data_files).isdisjoint(set(dataset.data_files))
