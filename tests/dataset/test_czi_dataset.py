import numpy as np
import pytest
from pylibCZIrw import czi as pyczi

from careamics.config import DataConfig
from careamics.dataset.czi_dataset import CZIDataset


@pytest.fixture
def dataset_czi(tmp_path):
    # Create a mock DataConfig
    data_config = DataConfig(
        data_type="czi",
        patch_size=[32, 32],
        axes="SYX",
    )

    n_files = 10
    files = []
    data_array = np.random.rand(10, 3, 64, 64).astype(np.float32)
    for i in range(n_files):
        file = tmp_path / f"sample{i}.czi"
        planes = [{"C": c, "T": 0, "Z": 0} for c in range(3)]

        with pyczi.create_czi(
            file, exist_ok=True, compression_options="uncompressed:"
        ) as czidoc_w:
            for plane in planes:

                czidoc_w.write(
                    data=data_array[i, plane["C"]],
                    plane=plane,
                )

        files.append(file)
    # Create the dataset instance
    return (
        CZIDataset(data_config=data_config, src_files=files, target_files=files),
        data_array,
    )


def test_calculate_mean_and_std(dataset_czi):

    image_stats, _ = dataset_czi[0]._calculate_mean_and_std()

    # Check if the means and stds are calculated
    assert image_stats.means is not None
    assert image_stats.stds is not None
    assert image_stats.means.shape == (3,)
    assert np.allclose(image_stats.means, dataset_czi[1].mean(axis=(0, 2, 3)))
    assert np.allclose(image_stats.stds, dataset_czi[1].std(axis=(0, 2, 3)))


def test_iter(dataset_czi):

    for patchs in dataset_czi[0]:
        assert isinstance(patchs[0], np.ndarray)
        assert isinstance(patchs[1], np.ndarray)
        assert patchs[0].shape == (3, 32, 32)
        assert patchs[1].shape == (3, 32, 32)


def test_get_number_of_files(dataset_czi):
    # Test the number of files
    assert dataset_czi[0].get_number_of_files() == len(dataset_czi[0].data_files)


def test_split_dataset(dataset_czi):
    # Test splitting the dataset
    split_dataset = dataset_czi[0].split_dataset(percentage=0.5, minimum_number=1)

    # Check that the number of files in the split dataset is correct
    assert split_dataset.get_number_of_files() == 5
    assert dataset_czi[0].get_number_of_files() == 5


def test_split_dataset_invalid_percentage(dataset_czi):
    # Test invalid percentage
    with pytest.raises(ValueError, match="Percentage must be between 0 and 1"):
        dataset_czi[0].split_dataset(percentage=1.5)


def test_split_dataset_invalid_minimum_number(dataset_czi):
    # Test invalid minimum number
    with pytest.raises(ValueError, match="Minimum number of files must be between 1"):
        dataset_czi[0].split_dataset(minimum_number=20)


if __name__ == "__main__":
    pytest.main()
