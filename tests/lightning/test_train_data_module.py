import numpy as np
import pytest
from tifffile import imwrite

from careamics.config import DataConfig
from careamics.config.support import (
    SupportedData,
    SupportedPixelManipulation,
    SupportedStructAxis,
)
from careamics.dataset import InMemoryDataset, PathIterableDataset
from careamics.lightning import TrainDataModule, create_train_datamodule


@pytest.fixture
def simple_array(ordered_array):
    return ordered_array((10, 10))


def test_mismatching_types_array(simple_array, minimum_data):
    """Test that an error is raised if the data type does not match the passed data."""
    minimum_data["data_type"] = SupportedData.TIFF.value
    with pytest.raises(ValueError):
        TrainDataModule(data_config=DataConfig(**minimum_data), train_data=simple_array)

    minimum_data["data_type"] = SupportedData.CUSTOM.value
    with pytest.raises(ValueError):
        TrainDataModule(data_config=DataConfig(**minimum_data), train_data=simple_array)

    minimum_data["data_type"] = SupportedData.ARRAY.value
    with pytest.raises(ValueError):
        TrainDataModule(
            data_config=DataConfig(**minimum_data), train_data="path/to/data"
        )


def test_wrapper_unknown_type(simple_array):
    """Test that an error is raised if the data type is not supported."""
    with pytest.raises(ValueError):
        create_train_datamodule(
            train_data=simple_array,
            data_type="wrong_type",
            patch_size=(10, 10),
            axes="YX",
            batch_size=2,
        )


def test_wrapper_train_array(simple_array):
    """Test that the data module is created correctly with an array."""
    # create data module
    data_module = create_train_datamodule(
        train_data=simple_array,
        data_type="array",
        patch_size=(8, 8),
        axes="YX",
        batch_size=2,
        val_minimum_patches=2,
    )
    data_module.prepare_data()
    data_module.setup()

    assert len(list(data_module.train_dataloader())) > 0


def test_wrapper_supervised_n2v_throws_error(simple_array):
    """Test that an error is raised if target data is passed but the transformations
    (default ones) contain N2V manipulate."""
    with pytest.raises(ValueError):
        create_train_datamodule(
            train_data=simple_array,
            data_type="array",
            patch_size=(10, 10),
            axes="YX",
            batch_size=2,
            train_target_data=simple_array,
            val_minimum_patches=2,
        )


@pytest.mark.parametrize(
    "use_n2v2, strategy",
    [
        (True, SupportedPixelManipulation.MEDIAN),
        (False, SupportedPixelManipulation.UNIFORM),
    ],
)
def test_wrapper_n2v2(simple_array, use_n2v2, strategy):
    """Test that n2v2 parameter is correctly passed."""
    data_module = create_train_datamodule(
        train_data=simple_array,
        data_type="array",
        patch_size=(16, 16),
        axes="YX",
        batch_size=2,
        use_n2v2=use_n2v2,
    )
    assert data_module.data_config.transforms[-1].strategy == strategy


def test_wrapper_structn2v(simple_array):
    """Test that structn2v parameter is correctly passed."""
    struct_axis = SupportedStructAxis.HORIZONTAL.value
    struct_span = 11

    data_module = create_train_datamodule(
        train_data=simple_array,
        data_type="array",
        patch_size=(16, 16),
        axes="YX",
        batch_size=2,
        struct_n2v_axis=struct_axis,
        struct_n2v_span=struct_span,
    )
    assert data_module.data_config.transforms[-1].struct_mask_axis == struct_axis
    assert data_module.data_config.transforms[-1].struct_mask_span == struct_span


def test_get_data_statistics(tmp_path):
    """Test that get_data_statistics works for every type of training dataset."""
    rng = np.random.default_rng(42)
    data = rng.integers(0, 255, (5, 3, 32, 32))
    data_mean = data.mean(axis=(0, 2, 3))
    data_std = data.std(axis=(0, 2, 3))

    data_val = rng.integers(0, 10, (2, 3, 32, 32))

    # create data module with in memory
    data_config = DataConfig(
        data_type=SupportedData.ARRAY.value,
        patch_size=(16, 16),
        axes="SCYX",
        batch_size=1,
    )
    data_module = TrainDataModule(
        data_config=data_config,
        train_data=data,
        val_data=data_val,
    )
    data_module.prepare_data()
    data_module.setup()
    assert isinstance(data_module.train_dataset, InMemoryDataset)

    means, stds = data_module.get_data_statistics()
    assert np.allclose(means, data_mean)
    assert np.allclose(stds, data_std)

    # same for path iterable
    train_path = tmp_path / "train"
    train_path.mkdir()
    val_path = tmp_path / "val"
    val_path.mkdir()

    for i in range(data.shape[0]):
        data_path = train_path / f"data_{i}.tif"
        imwrite(data_path, data[i])
    for i in range(data_val.shape[0]):
        data_path = val_path / f"data_{i}.tif"
        imwrite(data_path, data_val[i])

    data_config = DataConfig(
        data_type=SupportedData.TIFF.value,
        patch_size=(16, 16),
        axes="CYX",
        batch_size=1,
    )
    data_module = TrainDataModule(
        data_config=data_config,
        train_data=train_path,
        val_data=val_path,
        use_in_memory=False,
    )
    data_module.prepare_data()
    data_module.setup()
    assert isinstance(data_module.train_dataset, PathIterableDataset)

    means, stds = data_module.get_data_statistics()
    assert np.allclose(means, data_mean)
    assert np.allclose(stds, data_std)
