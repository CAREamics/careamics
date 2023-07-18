from pathlib import Path

import pytest

from careamics_restoration.config.data import Data, SupportedExtensions


@pytest.mark.parametrize("ext", ["npy", ".npy", "tiff", "tif", "TIFF", "TIF", ".TIF"])
def test_supported_extensions_case_insensitive(ext: str):
    """Test that SupportedExtension enum accepts all extensions in upper
    cases and with ."""
    sup_ext = SupportedExtensions(ext)

    new_ext = ext.lower()
    if ext.startswith("."):
        new_ext = new_ext[1:]

    assert sup_ext.value == new_ext


@pytest.mark.parametrize("ext", ["nd2", "jpg", "png ", "zarr"])
def test_wrong_extensions(minimum_config: dict, ext: str):
    """Test that supported model raises ValueError for unsupported extensions."""
    data_config = minimum_config["data"]
    data_config["data_format"] = ext

    # instantiate Data model
    with pytest.raises(ValueError):
        Data(**data_config)


@pytest.mark.parametrize("path", ["invalid/path", "training/file_0.tif"])
def test_invalid_training_path(tmp_path: Path, minimum_config: dict, path: str):
    """Test that Data model raises ValueError for invalid path."""
    data_config = minimum_config["data"]
    data_config["training_path"] = tmp_path / path

    # instantiate Data model
    with pytest.raises(ValueError):
        Data(**data_config)


def test_wrong_extension(minimum_config: dict):
    """Test that Data model raises ValueError for incompatible extension
    and path.

    Note: minimum configuration has data_format tif.
    """
    data_config = minimum_config["data"]
    data_config["data_format"] = "npy"

    # instantiate Data model
    with pytest.raises(ValueError):
        Data(**data_config)


def test_at_least_one_of_training_or_prediction(complete_config: dict):
    """Test that Data model raises an error if both training and prediction
    paths are None (not supplied)."""
    training = complete_config["data"]["training_path"]
    validation = complete_config["data"]["validation_path"]
    prediction = complete_config["data"]["prediction_path"]

    # None specified
    with pytest.raises(ValueError):
        Data(data_format="tif", axes="YX")

    # Only prediction specified
    data_model = Data(data_format="tif", axes="YX", prediction_path=prediction)
    assert str(data_model.prediction_path) == prediction

    # Only training specified
    data_model = Data(
        data_format="tif", axes="YX", training_path=training, validation_path=validation
    )
    assert str(data_model.training_path) == training


def test_both_training_and_validation(complete_config: dict):
    """Test that both training and validation paths must be specified together."""
    training = complete_config["data"]["training_path"]
    validation = complete_config["data"]["validation_path"]
    prediction = complete_config["data"]["prediction_path"]

    # None specified
    data_model = Data(data_format="tif", axes="YX", prediction_path=prediction)
    assert str(data_model.prediction_path) == prediction

    # Both specified
    data_model = Data(
        data_format="tif", axes="YX", training_path=training, validation_path=validation
    )
    assert str(data_model.training_path) == training
    assert str(data_model.validation_path) == validation

    # Only validation specified
    with pytest.raises(ValueError):
        Data(data_format="tif", axes="YX", validation_path=validation)

    # Only training specified
    with pytest.raises(ValueError):
        Data(data_format="tif", axes="YX", training_path=training)


@pytest.mark.parametrize("mean, std", [(0, 124.5), (12.6, 0.1)])
def test_mean_std_non_negative(complete_config: dict, mean, std):
    """Test that non negative mean and std are accepted."""
    complete_config["data"]["mean"] = mean
    complete_config["data"]["std"] = std

    data_model = Data(**complete_config["data"])
    assert data_model.mean == mean
    assert data_model.std == std


def test_mean_std_negative(complete_config: dict):
    """Test that negative mean and std are not accepted."""
    complete_config["data"]["mean"] = -1
    complete_config["data"]["std"] = 10.4

    with pytest.raises(ValueError):
        Data(**complete_config["data"])

    complete_config["data"]["mean"] = 10.4
    complete_config["data"]["std"] = -1

    with pytest.raises(ValueError):
        Data(**complete_config["data"])


def test_mean_std_both_specified_or_none(complete_config: dict):
    """Test an error is raised if std is specified but mean is None."""
    # No error if both are None
    complete_config["data"].pop("mean")
    complete_config["data"].pop("std")
    Data(**complete_config["data"])

    # No error if mean is defined
    complete_config["data"]["mean"] = 10.4
    Data(**complete_config["data"])

    # Error if only std is defined
    complete_config["data"].pop("mean")
    complete_config["data"]["std"] = 10.4

    with pytest.raises(ValueError):
        Data(**complete_config["data"])


def test_set_mean_and_std(complete_config: dict):
    """Test that mean and std can be set after initialization."""
    # they can be set both, when they are already set
    data = Data(**complete_config["data"])
    data.set_mean_and_std(4.07, 14.07)

    # and if they are both None
    complete_config["data"].pop("mean")
    complete_config["data"].pop("std")
    data = Data(**complete_config["data"])
    data.set_mean_and_std(10.4, 0.5)


def test_wrong_values_by_assigment(complete_config: dict):
    """Test that wrong values are not accepted through assignment."""
    data_model = Data(**complete_config["data"])

    # data format
    data_model.data_format = complete_config["data"]["data_format"]  # check assignment
    with pytest.raises(ValueError):
        data_model.data_format = "png"

    # axes
    data_model.axes = complete_config["data"]["axes"]
    with pytest.raises(ValueError):
        data_model.axes = "-YX"

    # training path
    data_model.training_path = complete_config["data"]["training_path"]
    with pytest.raises(ValueError):
        data_model.training_path = "invalid/path"

    # validation path
    data_model.validation_path = complete_config["data"]["validation_path"]
    with pytest.raises(ValueError):
        data_model.validation_path = "invalid/path"

    # prediction path
    data_model.prediction_path = complete_config["data"]["prediction_path"]
    with pytest.raises(ValueError):
        data_model.prediction_path = "invalid/path"

    # mean
    data_model.mean = complete_config["data"]["mean"]
    with pytest.raises(ValueError):
        data_model.mean = -1

    # std
    data_model.std = complete_config["data"]["std"]
    with pytest.raises(ValueError):
        data_model.std = -1


def test_data_to_dict_minimum(minimum_config: dict):
    """ "Test that export to dict does not include None values and Paths.

    In the minimum config, only training+validation should be defined, all the other
    paths are None."""
    data_minimum = Data(**minimum_config["data"]).model_dump()
    assert data_minimum == minimum_config["data"]

    assert "data_format" in data_minimum.keys()
    assert "axes" in data_minimum.keys()
    assert "training_path" in data_minimum.keys()
    assert "validation_path" in data_minimum.keys()
    assert "prediction_path" not in data_minimum.keys()


def test_data_to_dict_complete(complete_config: dict):
    """ "Test that export to dict does not include None values and Paths."""
    data_complete = Data(**complete_config["data"]).model_dump()
    assert data_complete == complete_config["data"]

    assert "data_format" in data_complete.keys()
    assert "axes" in data_complete.keys()
    assert "training_path" in data_complete.keys()
    assert "validation_path" in data_complete.keys()
    assert "prediction_path" in data_complete.keys()
