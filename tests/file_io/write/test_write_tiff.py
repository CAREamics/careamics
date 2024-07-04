import numpy as np
import pytest

from careamics.file_io.write import write_tiff


def test_write_tiff(tmp_path, ordered_array):
    """Test writing a tiff file."""
    # create an array
    array: np.ndarray = ordered_array((10, 10))

    # save files
    file = tmp_path / "test.tiff"
    write_tiff(file, array)

    assert file.is_file()


def test_invalid_extension_error(tmp_path, ordered_array):
    """Test error is raised when a path with an invalid extension is used."""
    # create an array
    array: np.ndarray = ordered_array((10, 10))

    # save files
    file = tmp_path / "test.invalid"
    with pytest.raises(ValueError):
        write_tiff(file, array)
