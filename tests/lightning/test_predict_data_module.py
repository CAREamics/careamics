import pytest

from careamics.config import InferenceConfig
from careamics.config.support import SupportedData
from careamics.lightning import PredictDataModule, create_predict_datamodule


@pytest.fixture
def simple_array(ordered_array):
    return ordered_array((10, 10))


def test_mismatching_types_array(simple_array, minimum_algorithm_n2v):
    """Test that an error is raised if the data type does not match the passed data."""
    minimum_algorithm_n2v["data_type"] = SupportedData.TIFF.value
    with pytest.raises(ValueError):
        PredictDataModule(
            data_config=InferenceConfig(**minimum_algorithm_n2v),
            train_data=simple_array,
        )

    minimum_algorithm_n2v["data_type"] = SupportedData.CUSTOM.value
    with pytest.raises(ValueError):
        PredictDataModule(
            data_config=InferenceConfig(**minimum_algorithm_n2v),
            train_data=simple_array,
        )

    minimum_algorithm_n2v["data_type"] = SupportedData.ARRAY.value
    with pytest.raises(ValueError):
        PredictDataModule(
            data_config=InferenceConfig(**minimum_algorithm_n2v),
            train_data="path/to/data",
        )


def test_wrapper_unknown_type(simple_array):
    """Test that an error is raised if the data type is not supported."""
    with pytest.raises(ValueError):
        create_predict_datamodule(
            pred_data=simple_array,
            data_type="wrong_type",
            image_means=[0.5],
            image_stds=[0.1],
            axes="YX",
            batch_size=2,
        )


def test_wrapper_instantiated_with_tiling(simple_array):
    """Test that the data module is created correctly with an array."""
    # create data module
    data_module = create_predict_datamodule(
        pred_data=simple_array,
        data_type="array",
        image_means=[0.5],
        image_stds=[0.1],
        axes="YX",
        batch_size=2,
        tile_overlap=[2, 2],
        tile_size=[8, 8],
    )

    data_module.prepare_data()
    data_module.setup()
    assert len(list(data_module.predict_dataloader())) == 2


def test_wrapper_instantiated_without_tiling(simple_array):
    """Test that the data module is created correctly with an array."""
    # create data module
    data_module = create_predict_datamodule(
        pred_data=simple_array,
        data_type="array",
        image_means=[0.5],
        image_stds=[0.1],
        axes="YX",
        batch_size=2,
    )

    data_module.prepare_data()
    data_module.setup()
    assert len(list(data_module.predict_dataloader())) == 1
