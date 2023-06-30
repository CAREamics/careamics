import pytest

from careamics_restoration.config.data import Data, SupportedExtensions


@pytest.mark.parametrize("ext", ["npy", ".npy", "tiff", "tif", "TIFF", "TIF", ".TIF"])
def test_supported_extensions_case_insensitive(ext):
    """Test that SupportedExtension enum accepts all extensions in upper
    cases and with ."""
    SupportedExtensions(ext)


@pytest.mark.parametrize("ext", ["nd2", "jpg", "png ", "zarr"])
def test_wrong_extensions(minimum_config, ext):
    """Test that supported model raises ValueError for unsupported extensions."""
    data_config = minimum_config["data"]
    data_config["data_format"] = ext

    # instantiate Data model
    with pytest.raises(ValueError):
        Data(**data_config)


@pytest.mark.parametrize("path", ["invalid/path", "training/file_0.tif"])
def test_invalid_training_path(tmp_path, minimum_config, path):
    """Test that Data model raises ValueError for invalid path."""
    data_config = minimum_config["data"]
    data_config["training_path"] = tmp_path / path

    # instantiate Data model
    with pytest.raises(ValueError):
        Data(**data_config)


def test_wrong_extension(minimum_config):
    """Test that Data model raises ValueError for incompatible extension
    and path.

    Note: minimum configuration has data_format tif.
    """
    data_config = minimum_config["data"]
    data_config["data_format"] = "npy"

    # instantiate Data model
    with pytest.raises(ValueError):
        Data(**data_config)
