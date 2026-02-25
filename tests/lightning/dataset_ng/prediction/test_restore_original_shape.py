import numpy as np
import pytest

from careamics.dataset.dataset_utils import reshape_array
from careamics.lightning.dataset_ng.prediction import restore_original_shape


def test_identity() -> None:
    """Test that `restore_original_shape` returns the same array if already in
    original shape."""
    original_shape = (5, 3, 32, 32)
    prediction = np.arange(np.prod(original_shape)).reshape(original_shape)
    restored = restore_original_shape(prediction, "SCYX", original_shape)
    assert np.array_equal(restored, prediction)


def test_singleton_s() -> None:
    """Test that `restore_original_shape` removes singleton S axis."""
    prediction = np.arange(3 * 32 * 32).reshape((1, 3, 32, 32))
    restored = restore_original_shape(prediction, "CYX", (3, 32, 32))
    assert np.array_equal(restored, prediction[0])


def test_singleton_c() -> None:
    """Test that `restore_original_shape` removes singleton C axis."""
    prediction = np.arange(5 * 32 * 32).reshape((5, 1, 32, 32))
    restored = restore_original_shape(prediction, "SYX", (5, 32, 32))
    assert np.array_equal(restored, prediction[:, 0])


def test_unflatten_s_and_t() -> None:
    """Test that `restore_original_shape` unflattens S axis to S and T."""
    prediction = np.arange(5 * 3 * 16 * 32 * 32).reshape((15, 1, 16, 32, 32))
    restored = restore_original_shape(prediction, "STZYX", (3, 5, 16, 32, 32))

    shape = restored.shape
    for s in range(shape[0]):
        for t in range(shape[1]):
            np.testing.assert_array_equal(
                restored[s, t], prediction[s * shape[1] + t, 0]
            )


def test_s_to_t() -> None:
    """Test that `restore_original_shape` converts S axis to T."""
    prediction = np.arange(5 * 32 * 32).reshape((5, 1, 32, 32))
    restored = restore_original_shape(prediction, "TYX", (5, 32, 32))

    shape = restored.shape
    for t in range(shape[0]):
        np.testing.assert_array_equal(restored[t], prediction[t, 0])


def test_unflatten_s_to_z_and_t() -> None:
    """Test that `restore_original_shape` unflattens S axis to Z and T.

    Case encountered with CZI when no depth axis is selected.
    """
    prediction = np.arange(5 * 3 * 32 * 32).reshape((15, 1, 32, 32))
    restored = restore_original_shape(prediction, "TZYX", (3, 5, 32, 32))

    shape = restored.shape
    for z in range(shape[0]):
        for t in range(shape[1]):
            np.testing.assert_array_equal(
                restored[z, t], prediction[z * shape[1] + t, 0]
            )


def test_z_to_t() -> None:
    """Test that `restore_original_shape` converts Z axis to T.

    Case encountered with CZI when T is selected as depth axis.
    """
    prediction = np.arange(15 * 32 * 32).reshape((1, 1, 15, 32, 32))
    restored = restore_original_shape(prediction, "TYX", (15, 32, 32))

    shape = restored.shape
    for t in range(shape[0]):
        np.testing.assert_array_equal(restored[t], prediction[0, 0, t])


def test_reorder_axes() -> None:
    """Test that `restore_original_shape` reorders axes to match original."""
    prediction = np.arange(5 * 3 * 16 * 32 * 32).reshape((5, 3, 16, 32, 32))
    restored = restore_original_shape(prediction, "CYXZS", (3, 32, 32, 16, 5))

    shape = restored.shape
    for c in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                for z in range(shape[3]):
                    for s in range(shape[4]):
                        np.testing.assert_array_equal(
                            restored[c, y, x, z, s], prediction[s, c, z, y, x]
                        )


@pytest.mark.parametrize(
    "shape, axes",
    [
        # identity, all axes
        ((5, 3, 32, 32), "SCYX"),
        ((5, 3, 16, 32, 32), "SCZYX"),
        # same order, not all axes
        ((32, 32), "YX"),
        ((3, 32, 32), "CYX"),
        ((16, 32, 32), "ZYX"),
        ((5, 32, 32), "SYX"),
        # with T
        ((8, 32, 32), "TYX"),
        ((8, 3, 32, 32), "TCYX"),
        ((8, 16, 32, 32), "TZYX"),
        # with S and T
        ((5, 8, 32, 32), "STYX"),
        ((8, 5, 32, 32), "TSYX"),
        # different order
        ((32, 32, 3), "YXC"),
        ((3, 32, 32, 16, 5), "CYXZS"),
    ],
)
def test_reshape_round_trip(shape, axes):
    original = np.arange(np.prod(shape)).reshape(shape)
    sczyx_array = reshape_array(original, axes)
    restored = restore_original_shape(sczyx_array, axes, shape)
    assert np.array_equal(restored, original)
