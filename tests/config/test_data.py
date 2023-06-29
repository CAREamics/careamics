from pathlib import Path

import pytest

from careamics_restoration.config.data import Data, SupportedExtension


def test_data(test_config):
    data_config = test_config["training"]["data"]

    # instantiate Data model
    data = Data(**data_config)

    # check that attributes are set correctly
    assert str(data.path) == data_config["path"]
    assert data.axes == data_config["axes"]
    assert data.ext == data_config["ext"]
    assert data.extraction_strategy == data_config["extraction_strategy"]
    assert data.batch_size == data_config["batch_size"]
    assert data.patch_size == data_config["patch_size"]

    # check that None optional attributes are not in the dictionary
    assert "num_files" not in data_config.keys()
    assert "num_patches" not in data_config.keys()
    assert "num_workers" not in data_config.keys()

    data_dict = data.dict()
    assert "num_files" not in data_dict
    assert "num_patches" not in data_dict

    # TODO default value is not None (revisit after cleaning up configuration)
    assert "num_workers" in data_dict

    # check that dictionary contains str not a Path
    assert isinstance(data_dict["path"], str)

    # add optional attributes
    path_to_data = Path(data_config["path"])
    extension = f"*.{data_config['ext']}"
    n_files = len(list(path_to_data.glob(extension)))

    data_config["num_files"] = n_files
    data_config["num_patches"] = 100
    data_config["num_workers"] = 8

    # instantiate Data model
    data = Data(**data_config)

    # check the optional attributes
    assert data.num_files == data_config["num_files"]
    assert data.num_patches == data_config["num_patches"]
    assert data.num_workers == data_config["num_workers"]

    # check that they are also in the dictionary
    data_dict = data.dict()
    assert data_dict["num_files"] == data_config["num_files"]
    assert data_dict["num_patches"] == data_config["num_patches"]
    assert data_dict["num_workers"] == data_config["num_workers"]


@pytest.mark.parametrize("ext", ["tiff", "tif", "TIFF", "TIF"])
def test_supported_extensions_case_insensitive(ext):
    """Test that SupportedExtension enum accepts all extensions in upper
    cases."""
    assert SupportedExtension(ext).value == ext.lower()


@pytest.mark.parametrize("ext", ["tiff", "tif", "TIFF", "TIF"])
def test_data_supported_extensions(test_config, ext):
    """Test that Data model accepts all extensions in upper cases."""
    data_config = test_config["training"]["data"]
    data_config["ext"] = ext

    # instantiate Data model
    data = Data(**data_config)
    assert data.ext == ext.lower()


@pytest.mark.parametrize("ext", ["nd2", "jpg", "png ", "zarr"])
def test_wrong_extensions(test_config, ext):
    """Test that Data model raises ValueError for unsupported extensions."""
    data_config = test_config["training"]["data"]
    data_config["ext"] = ext

    # instantiate Data model
    with pytest.raises(ValueError):
        Data(**data_config)


def test_invalid_path(test_config):
    """Test that Data model raises ValueError for invalid path."""
    data_config = test_config["training"]["data"]
    data_config["path"] = "invalid/path"

    # instantiate Data model
    with pytest.raises(ValueError):
        Data(**data_config)


def test_wrong_extension(test_config):
    """Test that Data model raises ValueError for incompatible extension
    and path."""
    data_config = test_config["training"]["data"]
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
def test_wrong_patch_and_axes(test_config, patch_size, axes):
    """Test that Data model raises ValueError for wrong axes."""
    data_config = test_config["training"]["data"]
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
def test_correct_patch_and_axes(test_config, patch_size, axes):
    """Test that Data model accepts correct axes."""
    data_config = test_config["training"]["data"]
    data_config["patch_size"] = list(patch_size)
    data_config["axes"] = axes

    # instantiate Data model
    data = Data(**data_config)
    assert data.axes == data_config["axes"]
    assert data.patch_size == data_config["patch_size"]
