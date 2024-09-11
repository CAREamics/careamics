import numpy as np
import pytest
import tifffile

from careamics.file_io.read import read_tiff


def test_read_tiff(tmp_path, ordered_array):
    """Test reading a tiff file."""
    # create an array
    array: np.ndarray = ordered_array((10, 10))

    # save files
    file = tmp_path / "test.tiff"
    tifffile.imwrite(file, array)

    # read files
    array_read = read_tiff(file)
    np.testing.assert_array_equal(array_read, array)


def test_read_tiff_invalid(tmp_path):
    # invalid file type
    file = tmp_path / "test.txt"
    file.write_text("test")
    with pytest.raises(ValueError):
        read_tiff(file)

    # non-existing file
    file = tmp_path / "test.tiff"
    with pytest.raises(FileNotFoundError):
        read_tiff(file)
