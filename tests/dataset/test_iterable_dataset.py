import numpy as np
import pytest
import tifffile

from careamics.config import DataConfig
from careamics.config.support import SupportedData
from careamics.dataset import PathIterableDataset
from careamics.dataset.dataset_utils import read_tiff


@pytest.mark.parametrize(
    "shape",
    [
        # 2D
        (32, 32),
        # 3D
        (32, 32, 32),
    ],
)
def test_number_of_files(tmp_path, ordered_array, shape):
    """Test number of files in PathIterableDataset."""
    # create array
    array_size = 32
    patch_size = 8
    n_files = 3
    factor = len(shape)
    axes = "YX" if factor == 2 else "ZYX"
    patch_sizes = [patch_size] * factor
    array = ordered_array(shape)

    # save three files
    files = []
    for i in range(n_files):
        file = tmp_path / f"array{i}.tif"
        tifffile.imwrite(file, array)
        files.append(file)

    # create config
    config_dict = {
        "data_type": SupportedData.TIFF.value,
        "patch_size": patch_sizes,
        "axes": axes,
    }
    config = DataConfig(**config_dict)

    # create dataset
    dataset = PathIterableDataset(
        data_config=config, src_files=files, read_source_func=read_tiff
    )

    # check number of files
    assert dataset.data_files == files

    # iterate over dataset
    patches = list(dataset)
    assert len(patches) == n_files * (array_size / patch_size) ** factor


def test_read_function(tmp_path, ordered_array):
    """Test reading files in PathIterableDataset using a custom read function."""

    # read function for .npy files
    def read_npy(file_path, *args, **kwargs):
        return np.load(file_path)

    array_size = 32
    patch_size = 8
    n_files = 3
    patch_sizes = [patch_size] * 2

    # create array
    array: np.ndarray = ordered_array((n_files, array_size, array_size))

    # save each plane in a single .npy file
    files = []
    for i in range(array.shape[0]):
        file_path = tmp_path / f"array{i}.npy"
        np.save(file_path, array[i])
        files.append(file_path)

    # create config
    config_dict = {
        "data_type": SupportedData.CUSTOM.value,
        "patch_size": patch_sizes,
        "axes": "YX",
    }
    config = DataConfig(**config_dict)

    # create dataset
    dataset = PathIterableDataset(
        data_config=config,
        src_files=files,
        read_source_func=read_npy,
    )
    assert dataset.data_files == files

    # iterate over dataset
    patches = list(dataset)
    assert len(patches) == n_files * (array_size / patch_size) ** 2


@pytest.mark.parametrize("percentage", [0.1, 0.6])
def test_extracting_val_files(tmp_path, ordered_array, percentage):
    """Test extracting a validation set patches from PathIterableDataset."""
    # create array
    array = ordered_array((20, 20))

    # save array to 25 files
    files = []
    for i in range(25):
        file_path = tmp_path / f"array{i}.tif"
        tifffile.imwrite(file_path, array)
        files.append(file_path)

    # create config
    config_dict = {
        "data_type": SupportedData.TIFF.value,
        "patch_size": [8, 8],
        "axes": "YX",
    }
    config = DataConfig(**config_dict)

    # create dataset
    dataset = PathIterableDataset(
        data_config=config, src_files=files, read_source_func=read_tiff
    )

    # compute number of patches
    total_n_files = dataset.get_number_of_files()
    minimum_files = 5
    n_files = max(round(percentage * total_n_files), minimum_files)

    # extract datset
    valset = dataset.split_dataset(percentage, minimum_files)

    # check number of patches
    assert valset.get_number_of_files() == n_files
    assert dataset.get_number_of_files() == total_n_files - n_files

    # check that none of the validation files are in the original dataset
    assert set(valset.data_files).isdisjoint(set(dataset.data_files))
