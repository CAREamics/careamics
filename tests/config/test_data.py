import pytest

from careamics_restoration.config.data import Data, SupportedExtensions

# TODO test fields


@pytest.mark.parametrize("ext", ["tiff", "tif", "TIFF", "TIF"])
def test_supported_extensions_case_insensitive(ext):
    """Test that SupportedExtension enum accepts all extensions in upper
    cases."""
    assert SupportedExtensions(ext).value == ext.lower()


@pytest.mark.parametrize("ext", ["tiff", "tif", "TIFF", "TIF"])
def test_data_supported_extensions(minimum_config, ext):
    """Test that Data model accepts all extensions in upper cases."""
    data_config = minimum_config["training"]["data"]
    data_config["ext"] = ext

    # instantiate Data model
    data = Data(**data_config)
    assert data.ext == ext.lower()


@pytest.mark.parametrize("ext", ["nd2", "jpg", "png ", "zarr"])
def test_wrong_extensions(minimum_config, ext):
    """Test that Data model raises ValueError for unsupported extensions."""
    data_config = minimum_config["training"]["data"]
    data_config["ext"] = ext

    # instantiate Data model
    with pytest.raises(ValueError):
        Data(**data_config)


def test_invalid_path(minimum_config):
    """Test that Data model raises ValueError for invalid path."""
    data_config = minimum_config["training"]["data"]
    data_config["path"] = "invalid/path"

    # instantiate Data model
    with pytest.raises(ValueError):
        Data(**data_config)


def test_wrong_extension(minimum_config):
    """Test that Data model raises ValueError for incompatible extension
    and path."""
    data_config = minimum_config["training"]["data"]
    data_config["ext"] = ".tiff"

    # instantiate Data model
    with pytest.raises(ValueError):
        Data(**data_config)


@pytest.mark.parametrize(
    "patch_size, axes",
    [
        # minimum 8
        ((2, 4), "YX"),
        ((8, 16, 4), "ZYX"),
        # power of two
        ((3, 8), "YX"),
        ((16, 18), "YX"),
        # wrong patch size
        ((8,), ("YX")),
        ((8, 16, 32, 64), "ZYX"),
        # wrong axes order
        ((8, 16), "XY"),
        ((8, 16, 32), "YXZ"),
        # missing YX
        ((8, 16, 32), "TZX"),
        # wrong axes length
        ((8, 16, 32), "YX"),
        # missing or supernumerary Z
        ((8, 16), "ZYX"),
        ((8, 16, 32), "TYX"),
        ((8, 16, 32), "SYX"),
    ],
)
def test_wrong_patch_and_axes(minimum_config, patch_size, axes):
    """Test that Data model raises ValueError for wrong axes."""
    data_config = minimum_config["training"]["data"]
    data_config["patch_size"] = list(patch_size)
    data_config["axes"] = axes

    # instantiate Data model
    with pytest.raises(ValueError):
        Data(**data_config)


@pytest.mark.parametrize(
    "patch_size, axes",
    [
        ((8, 16), "YX"),
        ((8, 16, 32), "ZYX"),
        ((8, 16), "TYX"),
        ((8, 16), "SYX"),
        ((8, 16, 32), "TZYX"),
        ((8, 16, 32), "SZYX"),
    ],
)
def test_correct_patch_and_axes(minimum_config, patch_size, axes):
    """Test that Data model accepts correct axes."""
    data_config = minimum_config["training"]["data"]
    data_config["patch_size"] = list(patch_size)
    data_config["axes"] = axes

    # instantiate Data model
    data = Data(**data_config)
    assert data.axes == data_config["axes"]
    assert data.patch_size == data_config["patch_size"]
