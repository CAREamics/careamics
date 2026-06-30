import numpy as np
import pytest

from careamics.utils.reshape_array import (
    get_stitch_slices,
    reshape_array,
    restore_array,
    restore_tile,
)
from tests.unit.utils.test_reshape_array import _DISORDERED_CASES, _ORDERED_CASES


@pytest.mark.parametrize("shape, axes", _ORDERED_CASES + _DISORDERED_CASES)
def test_restore_array_roundtrip(shape, axes):
    """Test that applying reshape then restore recovers the original array."""
    array = np.arange(np.prod(shape)).reshape(shape)
    reshaped = reshape_array(array, axes)
    restored = restore_array(reshaped, axes, shape)

    assert restored.shape == shape
    np.testing.assert_array_equal(restored, array)


@pytest.mark.parametrize("shape, axes", _ORDERED_CASES + _DISORDERED_CASES)
def test_indexing_tile(shape, axes):
    """Test that stich slices index tiles in the restored array."""
    array = np.arange(np.prod(shape)).reshape(shape)
    reshaped = reshape_array(array, axes)
    restored = restore_array(reshaped, axes, shape)

    # prepare tile, crop and coords
    if "Z" in axes:
        crop_size = (4, 8, 8)
        coords = (2, 2, 2)
    else:
        crop_size = (8, 8)
        coords = (2, 2)

    if "S" in axes or "T" in axes:
        idx_s = 2
    else:
        idx_s = 0

    tile = np.ones((reshaped.shape[1],) + crop_size)

    stitch_slices = get_stitch_slices(axes, shape, tile.shape, idx_s, coords, crop_size)
    restored_tile = restore_tile(tile, axes, shape)

    # check indexing
    restored[stitch_slices] = restored_tile
