import numpy as np
import pytest

from careamics.utils.reshape_array import (
    reshape_array,
    restore_array,
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
