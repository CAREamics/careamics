import pytest

from n2v.config.data import Data, SupportedExtension


@pytest.mark.parametrize("ext", ["tiff", "tif", "TIFF", "TIF"])
def test_supported_extensions_case_insensitive(ext):
    """Test that SupportedExtension enum accepts all extensions in upper
    cases."""
    assert SupportedExtension(ext).value == ext.lower()


@pytest.mark.parametrize("ext", ["tiff", "tif", "TIFF", "TIF"])
def test_data_supported_extensions(test_config, ext):
    data_config = test_config["training"]["data"]
    data_config["ext"] = ext

    # instantiate Data model
    data = Data(**data_config)
    assert data.ext == ext.lower()


@pytest.mark.parametrize("ext", ["npy", ".tif", "tifff", "zarr"])
def test_wrong_extensions(test_config, ext):
    data_config = test_config["training"]["data"]
    data_config["ext"] = ext

    # instantiate Data model
    with pytest.raises(ValueError):
        Data(**data_config)
