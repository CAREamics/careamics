import pytest

from careamics import CAREamicsTrainData, TrainingDataWrapper
from careamics.config import DataConfig
from careamics.config.support import (
    SupportedData,
    SupportedPixelManipulation,
    SupportedStructAxis,
)


@pytest.fixture
def simple_array(ordered_array):
    return ordered_array((10, 10))


def test_mismatching_types_array(simple_array, minimum_data):
    """Test that an error is raised if the data type does not match the passed data."""
    minimum_data["data_type"] = SupportedData.TIFF.value
    with pytest.raises(ValueError):
        CAREamicsTrainData(
            data_config=DataConfig(**minimum_data), train_data=simple_array
        )

    minimum_data["data_type"] = SupportedData.CUSTOM.value
    with pytest.raises(ValueError):
        CAREamicsTrainData(
            data_config=DataConfig(**minimum_data), train_data=simple_array
        )

    minimum_data["data_type"] = SupportedData.ARRAY.value
    with pytest.raises(ValueError):
        CAREamicsTrainData(
            data_config=DataConfig(**minimum_data), train_data="path/to/data"
        )


def test_wrapper_unknown_type(simple_array):
    """Test that an error is raised if the data type is not supported."""
    with pytest.raises(ValueError):
        TrainingDataWrapper(
            train_data=simple_array,
            data_type="wrong_type",
            patch_size=(10, 10),
            axes="YX",
            batch_size=2,
        )


def test_wrapper_train_array(simple_array):
    """Test that the data module is created correctly with an array."""
    # create data module
    data_module = TrainingDataWrapper(
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
        TrainingDataWrapper(
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
    data_module = TrainingDataWrapper(
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

    data_module = TrainingDataWrapper(
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
