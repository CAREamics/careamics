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

    # Only validation specified
    with pytest.raises(ValueError):
        Data(data_format="tif", axes="YX", validation_path=validation)

    # Only prediction specified
    data_model = Data(data_format="tif", axes="YX", prediction_path=prediction)
    assert str(data_model.prediction_path) == prediction

    # Only training specified
    data_model = Data(data_format="tif", axes="YX", training_path=training)
    assert str(data_model.training_path) == training


def test_data_to_dict_minimum(minimum_config: dict):
    """ "Test that export to dict does not include None values and Paths.

    In the minimum config, only training should be defined, all the other
    paths are None."""
    data_minimum = Data(**minimum_config["data"]).model_dump()
    assert data_minimum == minimum_config["data"]

    assert "data_format" in data_minimum.keys()
    assert "axes" in data_minimum.keys()
    assert "training_path" in data_minimum.keys()
    assert "validation_path" not in data_minimum.keys()
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
