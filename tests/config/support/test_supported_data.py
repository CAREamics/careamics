from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import pytest
import tifffile

from careamics.config.support import SupportedData


def test_extension_pattern_tiff_fnmatch(tmp_path: Path):
    """Test that the TIFF extension is compatible with fnmatch."""
    path = tmp_path / "test.tif"

    # test as str
    assert fnmatch(str(path), SupportedData.get_extension_pattern(SupportedData.TIFF))

    # test as Path
    assert fnmatch(path, SupportedData.get_extension_pattern(SupportedData.TIFF))


def test_extension_pattern_tiff_rglob(tmp_path: Path):
    """Test that the TIFF extension is compatible with Path.rglob."""
    # create text file
    text_path = tmp_path / "test.txt"
    text_path.write_text("test")

    # create image
    path = tmp_path / "test.tif"
    image = np.ones((10, 10))
    tifffile.imwrite(path, image)

    # search for files
    files = list(
        tmp_path.rglob(SupportedData.get_extension_pattern(SupportedData.TIFF))
    )
    assert len(files) == 1
    assert files[0] == path


def test_extension_pattern_custom_fnmatch(tmp_path: Path):
    """Test that the custom extension is compatible with fnmatch."""
    path = tmp_path / "test.czi"

    # test as str
    assert fnmatch(str(path), SupportedData.get_extension_pattern(SupportedData.CUSTOM))

    # test as Path
    assert fnmatch(path, SupportedData.get_extension_pattern(SupportedData.CUSTOM))


def test_extension_pattern_custom_rglob(tmp_path: Path):
    """Test that the custom extension is compatible with Path.rglob."""
    # create text file
    text_path = tmp_path / "test.txt"
    text_path.write_text("test")

    # create image
    path = tmp_path / "test.npy"
    image = np.ones((10, 10))
    np.save(path, image)

    # search for files
    files = list(
        tmp_path.rglob(SupportedData.get_extension_pattern(SupportedData.CUSTOM))
    )
    assert len(files) == 2
    assert set(files) == {path, text_path}


def test_extension_pattern_array_error():
    """Test that the array extension raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        SupportedData.get_extension_pattern(SupportedData.ARRAY)


def test_extension_pattern_any_error():
    """Test that any extension raises ValueError."""
    with pytest.raises(ValueError):
        SupportedData.get_extension_pattern("some random")


def test_extension_array_error():
    """Test that the array extension raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        SupportedData.get_extension(SupportedData.ARRAY)


def test_extension_tiff():
    """Test that the tiff extension is .tiff."""
    assert SupportedData.get_extension(SupportedData.TIFF) == ".tiff"


def test_extension_custom_error():
    """Test that the custom extension returns NotImplementedError."""
    with pytest.raises(NotImplementedError):
        SupportedData.get_extension(SupportedData.CUSTOM)


def test_extension_any_error():
    """Test that any extension raises ValueError."""
    with pytest.raises(ValueError):
        SupportedData.get_extension("some random")
