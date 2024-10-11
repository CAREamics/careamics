from unittest.mock import MagicMock

import numpy as np
import pytest

from careamics.file_io.read.czi_read import read_czi_roi, squeeze_possible


def test_squeeze_possible():
    # Test with a 2D array with shape (1, 5)
    array_1d = np.random.rand(1, 1, 5)
    result = squeeze_possible(array_1d)
    assert result.shape == (1, 5)  # Should squeeze the first dimension

    # Test with a 2D array with shape (3, 5)
    array_2d = np.random.rand(3, 5)
    result = squeeze_possible(array_2d)
    assert result.shape == (3, 5)  # Should not change the shape


@pytest.mark.parametrize(
    "patch_size",
    [
        (3, 2, 10, 10),
        (2, 1, 10, 10),
    ],
)
def test_read_czi_roi(patch_size):
    # Mock the CziReader and its read method
    mock_czi_reader = MagicMock()
    mock_czi_reader.read.return_value = np.random.rand(1, 10, 10, 1)  # Mocked data

    # Define patch size, coordinates, and plane
    cords = [0, 0, 0]  # (Z, X, Y)
    plane = {"C": 0}  # Assume a channel is specified

    # Call the function
    result = read_czi_roi("xy.czi", mock_czi_reader, patch_size, cords, plane)

    # Check the result shape
    assert result.shape[0] == patch_size[0]  # Should match the expected shape C
    # Should match the expected shape Z if Z!=1
    # else it squeezes the Z dimension to match the expected shape Y
    assert (
        result.shape[1] == patch_size[1]
        if patch_size[1] != 1
        else result.shape[1] == patch_size[2]
    )

    # # Test if read was called the expected number of times
    assert mock_czi_reader.read.call_count == (patch_size[0] * patch_size[1])


def test_read_czi_roi_invalid_patch_size():
    mock_czi_reader = MagicMock()
    patch_size = [2, 2, 10]  # Invalid (only 3 dimensions)
    cords = [0, 0, 0]
    plane = {"C": 0}

    with pytest.raises(ValueError, match="patch_size must have 4 dimensions"):
        read_czi_roi("xy.czi", mock_czi_reader, patch_size, cords, plane)


def test_read_czi_roi_invalid_cords():
    mock_czi_reader = MagicMock()
    patch_size = [2, 2, 10, 10]
    cords = [0, 0]  # Invalid (only 2 dimensions)
    plane = {"C": 0}

    with pytest.raises(ValueError, match="coords must have length 3"):
        read_czi_roi("xy.czi", mock_czi_reader, patch_size, cords, plane)


def test_read_czi_roi_none_plane():
    mock_czi_reader = MagicMock()
    patch_size = [2, 2, 10, 10]
    cords = [0, 0, 0]
    plane = None  # Invalid (None)

    with pytest.raises(ValueError, match="plane and coords cannot be None"):
        read_czi_roi("xy.czi", mock_czi_reader, patch_size, cords, plane)


def test_read_czi_roi_none_cords():
    mock_czi_reader = MagicMock()
    patch_size = [2, 2, 10, 10]
    cords = None  # Invalid (None)
    plane = {"C": 0}

    with pytest.raises(ValueError, match="plane and coords cannot be None"):
        read_czi_roi("xy.czi", mock_czi_reader, patch_size, cords, plane)
