import numpy as np
import pytest
import tifffile

from careamics.config import DataConfig
from careamics.config.support import SupportedData
from careamics.dataset import InMemoryDataset


def test_number_of_patches(ordered_array):
    """Test the number of patches extracted from InMemoryDataset."""
    # create array
    array = ordered_array((20, 20))

    # create config
    config_dict = {
        "data_type": SupportedData.ARRAY.value,
        "patch_size": [8, 8],
        "axes": "YX",
    }
    config = DataConfig(**config_dict)

    # create dataset
    dataset = InMemoryDataset(
        data_config=config,
        inputs=array,
    )

    # check number of patches
    assert len(dataset) == dataset.data.shape[0]


@pytest.mark.parametrize("percentage", [0.1, 0.6])
def test_extracting_val_array(ordered_array, percentage):
    """Test extracting a validation set patches from InMemoryDataset."""
    # create array
    array = ordered_array((32, 32))

    # create config
    config_dict = {
        "data_type": SupportedData.ARRAY.value,
        "patch_size": [8, 8],
        "axes": "YX",
    }
    config = DataConfig(**config_dict)

    # create dataset
    dataset = InMemoryDataset(
        data_config=config,
        inputs=array,
    )

    # compute number of patches
    total_n_patches = len(dataset)
    minimum_patches = 5
    n_patches = max(round(percentage * total_n_patches), minimum_patches)

    # extract datset
    valset = dataset.split_dataset(percentage, minimum_patches)

    # check number of patches
    assert len(valset) == n_patches
    assert len(dataset) == total_n_patches - n_patches

    # check that none of the validation patch values are in the original dataset
    assert np.in1d(valset.data, dataset.data).sum() == 0


@pytest.mark.parametrize("percentage", [0.1, 0.6])
def test_extracting_val_files(tmp_path, ordered_array, percentage):
    """Test extracting a validation set patches from InMemoryDataset."""
    # create array
    array = ordered_array((32, 32))

    # save array to file
    file_path = tmp_path / "array.tif"
    tifffile.imwrite(file_path, array)

    # create config
    config_dict = {
        "data_type": SupportedData.ARRAY.value,
        "patch_size": [8, 8],
        "axes": "YX",
    }
    config = DataConfig(**config_dict)

    # create dataset
    dataset = InMemoryDataset(
        data_config=config,
        inputs=[file_path],
    )

    # compute number of patches
    total_n_patches = len(dataset)
    minimum_patches = 5
    n_patches = max(round(percentage * total_n_patches), minimum_patches)

    # extract datset
    valset = dataset.split_dataset(percentage, minimum_patches)

    # check number of patches
    assert len(valset) == n_patches
    assert len(dataset) == total_n_patches - n_patches

    # check that none of the validation patch values are in the original dataset
    assert np.in1d(valset.data, dataset.data).sum() == 0
