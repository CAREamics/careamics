import pytest

import numpy as np
import tifffile

from careamics.config import DataModel
from careamics.config.support import SupportedData
from careamics.dataset import InMemoryDataset


def test_number_of_patches(ordered_array):
    """Test the number of patches extracted from InMemoryDataset."""
    # create array
    array = ordered_array((20, 20))

    # create config
    config_dict =  {
        "data_type": SupportedData.ARRAY.value,
        "patch_size": [4, 4],
        "axes": "YX",
    }
    config = DataModel(**config_dict)

    # create dataset
    dataset = InMemoryDataset(
        data_config=config,
        data=array,
    )

    # check number of patches
    assert dataset.get_number_of_patches() == dataset.patches.shape[0]


@pytest.mark.parametrize("percentage", [0.1, 0.6])
def test_extracting_val_array(ordered_array, percentage):
    """Test extracting a validation set patches from InMemoryDataset."""
    # create array
    array = ordered_array((20, 20))

    # create config
    config_dict =  {
        "data_type": SupportedData.ARRAY.value,
        "patch_size": [4, 4],
        "axes": "YX",
    }
    config = DataModel(**config_dict)

    # create dataset
    dataset = InMemoryDataset(
        data_config=config,
        data=array,
    )

    # compute number of patches
    total_n_patches = dataset.get_number_of_patches()
    minimum_patches = 5
    n_patches = max(round(percentage*total_n_patches), minimum_patches)

    # extract datset
    valset = dataset.split_dataset(percentage, minimum_patches)

    # check number of patches
    assert valset.get_number_of_patches() == n_patches
    assert dataset.get_number_of_patches() == total_n_patches - n_patches

    # check that none of the validation patch values are in the original dataset
    assert np.in1d(valset.patches, dataset.patches).sum() == 0


@pytest.mark.parametrize("percentage", [0.1, 0.6])
def test_extracting_val_files(tmp_path, ordered_array, percentage):
    """Test extracting a validation set patches from InMemoryDataset."""
    # create array
    array = ordered_array((20, 20))

    # save array to file
    file_path = tmp_path / "array.tif"
    tifffile.imwrite(file_path, array)

    # create config
    config_dict =  {
        "data_type": SupportedData.ARRAY.value,
        "patch_size": [4, 4],
        "axes": "YX",
    }
    config = DataModel(**config_dict)

    # create dataset
    dataset = InMemoryDataset(
        data_config=config,
        data=[file_path],
    )

    # compute number of patches
    total_n_patches = dataset.get_number_of_patches()
    minimum_patches = 5
    n_patches = max(round(percentage*total_n_patches), minimum_patches)

    # extract datset
    valset = dataset.split_dataset(percentage, minimum_patches)

    # check number of patches
    assert valset.get_number_of_patches() == n_patches
    assert dataset.get_number_of_patches() == total_n_patches - n_patches

    # check that none of the validation patch values are in the original dataset
    assert np.in1d(valset.patches, dataset.patches).sum() == 0
