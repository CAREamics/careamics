import numpy as np
import pytest

from careamics.utils.reshape_array import (
    AxesTransform,
    RestoredAxesTransform,
    reshape_array,
    restore_array,
    restore_tile,
)

# --- Test utilities


def _array_to_tile(input_shape, axes, target_shape):
    """Create the input to restore_tile tests from the restore_array test inputs."""
    return input_shape, axes, target_shape[1:]


def _transformed_shape(input_shape, axes):
    """Get the transformed shape from the restore_array test inputs."""
    transform = AxesTransform(axes, input_shape)
    return input_shape, axes, transform.transformed_shape


# default axes shapes
XY_S = 32
Z_S = 16
C_S = 3
T_S = 4
S_S = 5

# different output C dimension
OUT_C_S = 5

# Axes already in STCZYX order
_ORDERED_CASES = [
    ((XY_S, XY_S), "YX"),
    ((C_S, XY_S, XY_S), "CYX"),
    ((S_S, XY_S, XY_S), "SYX"),
    ((T_S, XY_S, XY_S), "TYX"),
    ((S_S, C_S, XY_S, XY_S), "SCYX"),
    ((T_S, C_S, XY_S, XY_S), "TCYX"),
    ((S_S, T_S, C_S, XY_S, XY_S), "STCYX"),
    ((Z_S, XY_S, XY_S), "ZYX"),
    ((C_S, Z_S, XY_S, XY_S), "CZYX"),
    ((S_S, Z_S, XY_S, XY_S), "SZYX"),
    ((T_S, Z_S, XY_S, XY_S), "TZYX"),
    ((S_S, C_S, Z_S, XY_S, XY_S), "SCZYX"),
    ((T_S, C_S, Z_S, XY_S, XY_S), "TCZYX"),
    ((S_S, T_S, C_S, Z_S, XY_S, XY_S), "STCZYX"),
]

# Axes in non-standard order
_DISORDERED_CASES = [
    ((XY_S, XY_S, C_S), "YXC"),
    ((XY_S, XY_S, Z_S), "YXZ"),
    ((XY_S, XY_S, T_S, Z_S), "YXTZ"),
    ((XY_S, XY_S, Z_S, C_S), "YXZC"),
    ((XY_S, XY_S, S_S), "YXS"),
    ((C_S, XY_S, XY_S, T_S, Z_S, S_S), "CYXTZS"),
]

# Ordered and disorded with transformed shape
_ORDERED_TRANSFORMED = [
    _transformed_shape(shape, axes) for shape, axes in _ORDERED_CASES
]

_DISORDERED_TRANSFORMED = [
    _transformed_shape(shape, axes) for shape, axes in _DISORDERED_CASES
]

_ORDERED_TRANSFORMED_TILE = [
    _array_to_tile(in_sh, axes, tar_sh) for in_sh, axes, tar_sh in _ORDERED_TRANSFORMED
]

_DISORDERED_TRANSFORMED_TILE = [
    _array_to_tile(in_sh, axes, tar_sh)
    for in_sh, axes, tar_sh in _DISORDERED_TRANSFORMED
]

# TODO test when input has singleton C
# Input-target with different C dimensions
_CHANNEL_REMOVED = [
    ((C_S, XY_S, XY_S), "CYX", (1, 1, XY_S, XY_S)),
    ((C_S, Z_S, XY_S, XY_S), "CZYX", (1, 1, Z_S, XY_S, XY_S)),
    ((S_S, C_S, XY_S, XY_S), "SCYX", (S_S, 1, XY_S, XY_S)),
    ((S_S, C_S, Z_S, XY_S, XY_S), "SCZYX", (S_S, 1, Z_S, XY_S, XY_S)),
    ((T_S, C_S, XY_S, XY_S), "TCYX", (T_S, 1, XY_S, XY_S)),
    ((T_S, C_S, Z_S, XY_S, XY_S), "TCZYX", (T_S, 1, Z_S, XY_S, XY_S)),
    ((S_S, T_S, C_S, XY_S, XY_S), "STCYX", (S_S * T_S, 1, XY_S, XY_S)),
    ((S_S, T_S, C_S, Z_S, XY_S, XY_S), "STCZYX", (S_S * T_S, 1, Z_S, XY_S, XY_S)),
]

_CHANNEL_ADDED = [
    ((XY_S, XY_S), "YX", (1, C_S, XY_S, XY_S)),
    ((Z_S, XY_S, XY_S), "ZYX", (1, C_S, Z_S, XY_S, XY_S)),
    ((S_S, XY_S, XY_S), "SYX", (S_S, C_S, XY_S, XY_S)),
    ((S_S, Z_S, XY_S, XY_S), "SZYX", (S_S, C_S, Z_S, XY_S, XY_S)),
    ((T_S, XY_S, XY_S), "TYX", (T_S, C_S, XY_S, XY_S)),
    ((T_S, Z_S, XY_S, XY_S), "TZYX", (T_S, C_S, Z_S, XY_S, XY_S)),
    ((S_S, T_S, XY_S, XY_S), "STYX", (S_S * T_S, C_S, XY_S, XY_S)),
    ((S_S, T_S, Z_S, XY_S, XY_S), "STZYX", (S_S * T_S, C_S, Z_S, XY_S, XY_S)),
]

_CHANNEL_CHANGED = [
    ((C_S, XY_S, XY_S), "CYX", (1, OUT_C_S, XY_S, XY_S)),
    ((C_S, Z_S, XY_S, XY_S), "CZYX", (1, OUT_C_S, Z_S, XY_S, XY_S)),
    ((S_S, C_S, XY_S, XY_S), "SCYX", (S_S, OUT_C_S, XY_S, XY_S)),
    ((S_S, C_S, Z_S, XY_S, XY_S), "SCZYX", (S_S, OUT_C_S, Z_S, XY_S, XY_S)),
    ((T_S, C_S, XY_S, XY_S), "TCYX", (T_S, OUT_C_S, XY_S, XY_S)),
    ((S_S, T_S, C_S, XY_S, XY_S), "STCYX", (S_S * T_S, OUT_C_S, XY_S, XY_S)),
]

_CHANNEL_MISMATCH = _CHANNEL_REMOVED + _CHANNEL_ADDED + _CHANNEL_CHANGED

_CHANNEL_REMOVED_DISORDERED = [
    ((XY_S, XY_S, C_S), "YXC", (1, 1, XY_S, XY_S)),
    ((XY_S, XY_S, C_S, Z_S), "YXCZ", (1, 1, Z_S, XY_S, XY_S)),
    ((XY_S, XY_S, C_S, T_S), "YXCT", (T_S, 1, XY_S, XY_S)),
    ((XY_S, XY_S, T_S, C_S, Z_S), "YXTCZ", (T_S, 1, Z_S, XY_S, XY_S)),
    ((XY_S, XY_S, T_S, C_S, S_S, Z_S), "YXTCSZ", (S_S * T_S, 1, Z_S, XY_S, XY_S)),
]

_CHANNEL_ADDED_DISORDERED = [
    ((XY_S, XY_S, Z_S), "YXZ", (1, C_S, Z_S, XY_S, XY_S)),
    ((XY_S, XY_S, T_S), "YXT", (T_S, C_S, XY_S, XY_S)),
    ((XY_S, XY_S, T_S, Z_S), "YXTZ", (T_S, C_S, Z_S, XY_S, XY_S)),
    ((XY_S, XY_S, T_S, S_S, Z_S), "YXTSZ", (S_S * T_S, C_S, Z_S, XY_S, XY_S)),
]


_CHANNEL_CHANGED_DISORDERED = [
    ((XY_S, XY_S, C_S), "YXC", (1, OUT_C_S, XY_S, XY_S)),
    ((XY_S, XY_S, C_S, Z_S), "YXCZ", (1, OUT_C_S, Z_S, XY_S, XY_S)),
    ((C_S, XY_S, XY_S, T_S, S_S), "CYXTS", (S_S * T_S, OUT_C_S, XY_S, XY_S)),
    ((C_S, XY_S, XY_S, T_S, Z_S, S_S), "CYXTZS", (S_S * T_S, OUT_C_S, Z_S, XY_S, XY_S)),
]

_CHANNEL_MISMATCH_DISORDERED = (
    _CHANNEL_REMOVED_DISORDERED
    + _CHANNEL_ADDED_DISORDERED
    + _CHANNEL_CHANGED_DISORDERED
)

_CHANNEL_MISMATCH_TILE = [
    _array_to_tile(in_sh, axes, tar_sh) for in_sh, axes, tar_sh in _CHANNEL_MISMATCH
]

_CHANNEL_MISMATCH_TILE_DISORDERED = [
    _array_to_tile(in_sh, axes, tar_sh)
    for in_sh, axes, tar_sh in _CHANNEL_MISMATCH_DISORDERED
]


# --- Unit tests


class TestAxesTransform:
    def test_s_added(self):
        t = AxesTransform("YX", (XY_S, XY_S))
        assert t.c_added_to_original is True
        assert len(t.sample_dims) == 0

    def test_t_becomes_s(self):
        t = AxesTransform("TYX", (T_S, XY_S, XY_S))
        assert len(t.sample_dims) == 1
        assert t.sample_dims[0] == "T"

    def test_st_merged(self):
        t = AxesTransform("STYX", (S_S, T_S, XY_S, XY_S))
        assert len(t.sample_dims) == 2
        assert set(t.sample_dims) == {"S", "T"}
        assert t.c_added_to_original is True

    def test_c_added(self):
        t = AxesTransform("SYX", (S_S, XY_S, XY_S))
        assert t.c_added_to_original is True

    def test_c_not_added(self):
        t = AxesTransform("SCYX", (S_S, C_S, XY_S, XY_S))
        assert t.c_added_to_original is False

    def test_has_z(self):
        assert AxesTransform("ZYX", (Z_S, XY_S, XY_S)).original_has_z is True
        assert AxesTransform("YX", (XY_S, XY_S)).original_has_z is False

    def test_dl_axes_2d(self):
        assert AxesTransform("YX", (XY_S, XY_S)).transformed_axes == "SCYX"

    def test_dl_axes_3d(self):
        assert AxesTransform("ZYX", (Z_S, XY_S, XY_S)).transformed_axes == "SCZYX"

    def test_dl_shape_yx(self):
        assert AxesTransform("YX", (XY_S, XY_S)).transformed_shape == (1, 1, XY_S, XY_S)

    def test_dl_shape_with_c(self):
        assert AxesTransform("YXC", (XY_S, XY_S, C_S)).transformed_shape == (
            1,
            C_S,
            XY_S,
            XY_S,
        )

    def test_dl_shape_with_st(self):
        transform = AxesTransform("STCYX", (S_S, T_S, C_S, XY_S, XY_S))
        assert transform.transformed_shape == (S_S * T_S, C_S, XY_S, XY_S)

    def test_dl_shape_t_as_s(self):
        transform = AxesTransform("TYX", (T_S, XY_S, XY_S))
        assert transform.transformed_shape == (T_S, 1, XY_S, XY_S)

    def test_invalid_axis_name(self):
        with pytest.raises(ValueError):
            AxesTransform("ABX", (1, 2, 3))

    def test_duplicate_axes(self):
        with pytest.raises(ValueError):
            AxesTransform("YYX", (32, 32, 32))

    def test_missing_y_or_x(self):
        with pytest.raises(ValueError):
            AxesTransform("SC", (5, 3))

    def test_shape_axes_length_mismatch(self):
        with pytest.raises(ValueError):
            AxesTransform("YX", (32, 32, 32))

    @pytest.mark.parametrize("shape, axes", _ORDERED_CASES + _DISORDERED_CASES)
    def test_reshape_array_dl_axes(self, shape, axes):
        """Result should always have S, C, and optionally Z, then Y, X."""
        transform = AxesTransform(axes, shape)
        expected_axes = "SCZYX" if "Z" in axes else "SCYX"
        assert transform.transformed_axes == expected_axes


class TestReshape:

    @pytest.mark.parametrize("shape, axes", _ORDERED_CASES + _DISORDERED_CASES)
    def test_reshape_array_produces_correct_shape(self, shape, axes):
        array = np.zeros(shape)
        result = reshape_array(array, axes)
        transform = AxesTransform(axes, shape)

        assert result.shape == transform.transformed_shape
        assert result.ndim in (4, 5)

    @pytest.mark.parametrize(
        "shape, axes, expected_s",
        [
            ((XY_S, XY_S), "YX", 1),  # singleton S added
            ((S_S, XY_S, XY_S), "SYX", 5),  # S preserved
            ((T_S, XY_S, XY_S), "TYX", 4),  # T becomes S
            ((S_S, T_S, C_S, XY_S, XY_S), "STCYX", 20),  # S*T merged
        ],
    )
    def test_reshape_array_s_dimension(self, shape, axes, expected_s):
        array = np.zeros(shape)
        result = reshape_array(array, axes)
        assert result.shape[0] == expected_s

    @pytest.mark.parametrize(
        "shape, axes, expected_c",
        [
            ((XY_S, XY_S), "YX", 1),  # singleton C added
            ((C_S, XY_S, XY_S), "CYX", 3),  # C preserved
        ],
    )
    def test_reshape_array_c_dimension(self, shape, axes, expected_c):
        array = np.zeros(shape)
        result = reshape_array(array, axes)
        assert result.shape[1] == expected_c

    def test_identity(self) -> None:
        """Test that `reshape_array` returns the same array if already in
        original shape."""
        original_axes = "SCYX"
        original_shape = (S_S, C_S, XY_S, XY_S)

        array = np.arange(np.prod(original_shape)).reshape(original_shape)
        restored = reshape_array(array, original_axes)
        assert np.array_equal(restored, array)

    def test_singleton_s(self) -> None:
        """Test that `reshape_array` adds singleton S axis."""
        original_axes = "CYX"
        original_shape = (C_S, XY_S, XY_S)

        array = np.arange(np.prod(original_shape)).reshape(original_shape)
        restored = reshape_array(array, original_axes)
        assert np.array_equal(restored[0], array)

    def test_singleton_c(self) -> None:
        """Test that `reshape_array` adds singleton C axis."""
        original_axes = "SYX"
        original_shape = (S_S, XY_S, XY_S)

        array = np.arange(np.prod(original_shape)).reshape(original_shape)
        restored = reshape_array(array, original_axes)
        assert np.array_equal(restored[:, 0, ...], array)

    def test_unflatten_s_and_t(self) -> None:
        """Test that `reshape_array` merges S and T into S."""
        original_axes = "STYX"
        original_shape = (S_S, T_S, XY_S, XY_S)

        array = np.arange(np.prod(original_shape)).reshape(original_shape)
        restored = reshape_array(array, original_axes)

        for s in range(original_shape[0]):
            for t in range(original_shape[1]):
                np.testing.assert_array_equal(
                    restored[s * original_shape[1] + t, 0], array[s, t]
                )

    def test_s_to_t(self) -> None:
        """Test that `reshape_array` converts S axis to T."""
        original_axes = "TYX"
        original_shape = (T_S, XY_S, XY_S)

        array = np.arange(np.prod(original_shape)).reshape(original_shape)
        restored = reshape_array(array, original_axes)
        np.testing.assert_array_equal(restored[:, 0, ...], array)

    def test_reorder_axes(self) -> None:
        """Test that `reshape_array` reorders axes to match original."""
        original_axes = "CYXZS"
        original_shape = (C_S, XY_S, XY_S, Z_S, S_S)

        array = np.arange(np.prod(original_shape)).reshape(original_shape)
        restored = reshape_array(array, original_axes)

        for c in range(original_shape[0]):
            for z in range(original_shape[3]):
                for s in range(original_shape[4]):
                    np.testing.assert_array_equal(
                        restored[s, c, z], array[c, :, :, z, s]
                    )


# TODO currently testing the wrapper function rather than the underlying class
class TestRestoreArray:
    """Test that restore_array specifically put the values in the right dimensions."""

    def test_identity(self) -> None:
        """Test that `restore_array` returns the same array if already in
        original shape."""
        original_shape = (5, 3, 32, 32)
        prediction = np.arange(np.prod(original_shape)).reshape(original_shape)
        restored = restore_array(prediction, "SCYX", original_shape)
        assert np.array_equal(restored, prediction)

    def test_singleton_s(self) -> None:
        """Test that `restore_array` removes singleton S axis."""
        prediction = np.arange(3 * 32 * 32).reshape((1, 3, 32, 32))
        restored = restore_array(prediction, "CYX", (3, 32, 32))
        assert np.array_equal(restored, prediction[0])

    def test_singleton_c(self) -> None:
        """Test that `restore_array` removes singleton C axis."""
        prediction = np.arange(5 * 32 * 32).reshape((5, 1, 32, 32))
        restored = restore_array(prediction, "SYX", (5, 32, 32))
        assert np.array_equal(restored, prediction[:, 0])

    def test_unflatten_s_and_t(self) -> None:
        """Test that `restore_array` unflattens S axis to S and T."""
        prediction = np.arange(5 * 3 * 16 * 32 * 32).reshape((15, 1, 16, 32, 32))
        restored = restore_array(prediction, "STZYX", (3, 5, 16, 32, 32))

        shape = restored.shape
        for s in range(shape[0]):
            for t in range(shape[1]):
                np.testing.assert_array_equal(
                    restored[s, t], prediction[s * shape[1] + t, 0]
                )

    def test_s_to_t(self) -> None:
        """Test that `restore_array` converts S axis to T."""
        prediction = np.arange(5 * 32 * 32).reshape((5, 1, 32, 32))
        restored = restore_array(prediction, "TYX", (5, 32, 32))

        shape = restored.shape
        for t in range(shape[0]):
            np.testing.assert_array_equal(restored[t], prediction[t, 0])

    def test_reorder_axes(self) -> None:
        """Test that `restore_array` reorders axes to match original."""
        prediction = np.arange(5 * 3 * 16 * 32 * 32).reshape((5, 3, 16, 32, 32))
        restored = restore_array(prediction, "CYXZS", (3, 32, 32, 16, 5))

        shape = restored.shape
        for c in range(shape[0]):
            for y in range(shape[1]):
                for x in range(shape[2]):
                    for z in range(shape[3]):
                        for s in range(shape[4]):
                            np.testing.assert_array_equal(
                                restored[c, y, x, z, s], prediction[s, c, z, y, x]
                            )

    def test_restore_array_wrong_ndim(self):
        with pytest.raises(ValueError, match="Expected 4D"):
            restore_array(np.zeros((32, 32, 32)), "YX", (32, 32))


# TODO currently testing the wrapper function rather than the underlying class
class TestRestoreTile:
    @pytest.mark.parametrize(
        "original_shape, original_axes, tile_shape",
        [
            # 2D cases — tile is CYX
            ((XY_S, XY_S), "YX", (1, XY_S, XY_S)),
            ((C_S, XY_S, XY_S), "CYX", (C_S, XY_S, XY_S)),
            ((XY_S, XY_S, C_S), "YXC", (C_S, XY_S, XY_S)),
            ((S_S, XY_S, XY_S), "SYX", (1, XY_S, XY_S)),
            ((T_S, XY_S, XY_S), "TYX", (1, XY_S, XY_S)),
            ((S_S, C_S, XY_S, XY_S), "SCYX", (C_S, XY_S, XY_S)),
            ((S_S, T_S, C_S, XY_S, XY_S), "STCYX", (C_S, XY_S, XY_S)),
            # 3D cases — tile is CZYX
            ((Z_S, XY_S, XY_S), "ZYX", (1, Z_S, XY_S, XY_S)),
            ((C_S, Z_S, XY_S, XY_S), "CZYX", (C_S, Z_S, XY_S, XY_S)),
            ((S_S, C_S, Z_S, XY_S, XY_S), "SCZYX", (C_S, Z_S, XY_S, XY_S)),
            ((S_S, T_S, C_S, Z_S, XY_S, XY_S), "STCZYX", (C_S, Z_S, XY_S, XY_S)),
        ],
    )
    def test_restore_tile(self, original_shape, original_axes, tile_shape):
        tile = np.random.rand(*tile_shape)
        restored = restore_tile(tile, original_axes, original_shape)

        # expected tile axes: original without S and T
        expected_axes = "".join(a for a in original_axes if a not in "ST")

        assert restored.ndim == len(expected_axes)

        # spatial dims should be preserved
        for ax in "YX":
            orig_idx = expected_axes.index(ax)
            assert restored.shape[orig_idx] == original_shape[original_axes.index(ax)]

    def test_restore_tile_reorder(self):
        """Tile from YXC data should restore C to last position."""
        tile = np.arange(C_S * XY_S * XY_S).reshape(C_S, XY_S, XY_S)  # CYX in DL space
        restored = restore_tile(tile, "YXC", (XY_S, XY_S, C_S))
        assert restored.shape == (XY_S, XY_S, C_S)
        assert restored[2, 3, 1] == tile[1, 2, 3]

    def test_restore_tile_singleton_c_removed(self):
        """Tile from YX data (no C) should have C squeezed out."""
        tile = np.random.rand(1, XY_S, XY_S)  # CYX with C=1
        restored = restore_tile(tile, "YX", (XY_S, XY_S))
        assert restored.shape == (XY_S, XY_S)

    def test_restore_tile_wrong_ndim(self):
        with pytest.raises(ValueError, match="Expected 3D"):
            restore_tile(np.zeros((XY_S, XY_S)), "YX", (XY_S, XY_S))


class TestRestoredAxesTransform:
    """Test class restoring array, tile and stitch slices."""

    # test errors
    @pytest.mark.parametrize(
        "axes, shape, target_shape, is_tile, exp_error",
        [
            (
                "CYX",
                (C_S, XY_S, XY_S),
                (XY_S, XY_S),
                True,
                pytest.raises(ValueError, match="is not a valid"),
            ),
            (
                "CYX",
                (C_S, XY_S, XY_S),
                (C_S, XY_S, XY_S),
                False,
                pytest.raises(ValueError, match="is not a valid array"),
            ),
        ],
    )
    def test_current_shape_mismatch_error(
        self, axes, shape, target_shape, is_tile, exp_error
    ):
        """Test that a mismatch between the length of the current shape and whether the
        array is a tile raises an error."""
        with exp_error:
            RestoredAxesTransform(axes, shape, target_shape, is_tile)

    @pytest.mark.parametrize(
        "axes, shape, target_shape, is_tile, exp_error",
        [
            (
                "CYX",
                (C_S, XY_S, XY_S),
                (C_S, XY_S, XY_S, XY_S),
                True,
                pytest.raises(ValueError, match="must both contain Z or neither"),
            ),
            (
                "CZYX",
                (C_S, Z_S, XY_S, XY_S),
                (C_S, XY_S, XY_S),
                True,
                pytest.raises(ValueError, match="must both contain Z or neither"),
            ),
        ],
    )
    def test_spatial_mismatch_error(
        self, axes, shape, target_shape, is_tile, exp_error
    ):
        """Test that a mismatch between the spatial dimensions of the current shape and
        the original shape raises an error."""
        with exp_error:
            RestoredAxesTransform(axes, shape, target_shape, is_tile)

    @pytest.mark.parametrize(
        "in_shape, axes, target_shape",
        _ORDERED_TRANSFORMED + _DISORDERED_TRANSFORMED,
    )
    def test_restored_array_shape(self, in_shape, axes, target_shape):
        """Test that the restored array shape is the original shape in the absence of
        channel mismatch."""
        transform = RestoredAxesTransform(
            axes, in_shape, target_shape, current_is_tile=False
        )
        # test that the restored array shape is the original shape
        assert transform.restored_array_shape == in_shape

    @pytest.mark.parametrize(
        "in_shape, axes, target_shape",
        _ORDERED_TRANSFORMED
        + _DISORDERED_TRANSFORMED
        + _CHANNEL_CHANGED
        + _CHANNEL_CHANGED_DISORDERED,
    )
    def test_restored_array_axes(self, in_shape, axes, target_shape):
        """Test that the restored array axes are the original axes in the absence of
        channel mismatch."""
        transform = RestoredAxesTransform(
            axes, in_shape, target_shape, current_is_tile=False
        )
        # test that the restored array axes are the original axes
        assert "".join(transform.restored_array_axes) == axes

    # TODO should this property kick out singleton C if they were present in input?
    @pytest.mark.parametrize(
        "in_shape, axes, target_shape",
        _CHANNEL_REMOVED
        + _CHANNEL_ADDED
        + _CHANNEL_REMOVED_DISORDERED
        + _CHANNEL_ADDED_DISORDERED,
    )
    def test_restored_array_axes_channel_mismatch(self, in_shape, axes, target_shape):
        """Test that the restored array axes include C when it is not singleton in the
        target shape when there is a channel mismatch."""
        transform = RestoredAxesTransform(
            axes, in_shape, target_shape, current_is_tile=False
        )
        new_axes = "".join(transform.restored_array_axes)

        # test that C is included in the restored axes if it is not singleton in target
        assert ("C" in new_axes) == (target_shape[1] > 1)

        # remove C from either and test that they are the same
        purged_axes = "".join(ax for ax in new_axes if ax != "C")
        purged_target_axes = "".join(ax for ax in axes if ax != "C")
        assert purged_axes == purged_target_axes

    @pytest.mark.parametrize(
        "in_shape, axes, target_shape",
        _ORDERED_TRANSFORMED
        + _DISORDERED_TRANSFORMED
        + _CHANNEL_MISMATCH
        + _CHANNEL_MISMATCH_DISORDERED,
    )
    def test_current_c_size(self, in_shape, axes, target_shape):
        """Test that the current channel is correct."""
        transform = RestoredAxesTransform(
            axes, in_shape, target_shape, current_is_tile=False
        )
        assert transform.current_c_size == target_shape[1]

    # TODO combine with previous test
    @pytest.mark.parametrize(
        "in_shape, axes, target_shape",
        _ORDERED_TRANSFORMED_TILE
        + _DISORDERED_TRANSFORMED_TILE
        + _CHANNEL_MISMATCH_TILE
        + _CHANNEL_MISMATCH_TILE_DISORDERED,
    )
    def test_current_c_size_tile(self, in_shape, axes, target_shape):
        """Test that the current channel is correct."""
        transform = RestoredAxesTransform(
            axes, in_shape, target_shape, current_is_tile=True
        )
        assert transform.current_c_size == target_shape[0]

    @pytest.mark.parametrize(
        "in_shape, axes, target_shape",
        _ORDERED_TRANSFORMED + _DISORDERED_TRANSFORMED,
    )
    def test_restore_array(self, in_shape, axes, target_shape):
        """Test that the restored array shape is the original shape in the absence of
        channel mismatch."""
        transform = RestoredAxesTransform(
            axes, in_shape, target_shape, current_is_tile=False
        )
        restored = transform.restore(np.zeros(target_shape))

        # test that the restored array shape is the original shape
        assert restored.shape == in_shape

    # TODO combine with previous test
    @pytest.mark.parametrize(
        "in_shape, axes, target_shape",
        _ORDERED_TRANSFORMED_TILE + _DISORDERED_TRANSFORMED_TILE,
    )
    def test_restore_tile(self, in_shape, axes, target_shape):
        """Test that the restored array shape is the original shape in the absence of
        channel mismatch."""
        transform = RestoredAxesTransform(
            axes, in_shape, target_shape, current_is_tile=True
        )
        restored = transform.restore(np.zeros(target_shape))

        # test that the restored array shape is the original shape
        tile_shape = tuple(
            in_shape[axes.index(ax)] for ax in axes if ax not in ("S", "T")
        )
        assert restored.shape == tile_shape

    @pytest.mark.parametrize(
        "in_shape, axes, target_shape",
        _CHANNEL_MISMATCH + _CHANNEL_MISMATCH_DISORDERED,
    )
    def test_restore_array_channel_mismatch(self, in_shape, axes, target_shape):
        """Test that the restored array shape is correct in the presence of channel
        mismatch."""
        transform = RestoredAxesTransform(
            axes, in_shape, target_shape, current_is_tile=False
        )
        restored = transform.restore(np.zeros(target_shape))

        # test that the only difference between in_shape and restored.shape is the
        # channel dimension
        same_dims = len(in_shape) == len(restored.shape)
        if same_dims:
            # C dim is that of target_shape
            assert restored.shape[axes.index("C")] == target_shape[1]
        else:
            c_removed = len(in_shape) == len(restored.shape) + 1
            set_in_shape = set(in_shape)
            set_restored_shape = set(restored.shape)

            if c_removed:
                # C dim is removed from restored.shape, difference should be original C
                assert set_in_shape - set_restored_shape == {in_shape[axes.index("C")]}
            else:
                # C dim is added to restored.shape, difference should be target C
                assert set_restored_shape - set_in_shape == {target_shape[1]}

    # TODO combine with previous test
    @pytest.mark.parametrize(
        "in_shape, axes, target_shape",
        _CHANNEL_MISMATCH_TILE + _CHANNEL_MISMATCH_TILE_DISORDERED,
    )
    def test_restore_tile_channel_mismatch(self, in_shape, axes, target_shape):
        """Test that the restored array shape is correct in the presence of channel
        mismatch."""
        transform = RestoredAxesTransform(
            axes, in_shape, target_shape, current_is_tile=True
        )
        restored = transform.restore(np.zeros(target_shape))

        # test that the only difference between in_shape and restored.shape is the
        # channel dimension
        in_shape_tile = tuple(
            in_shape[axes.index(ax)] for ax in axes if ax not in ("S", "T")
        )
        new_axes = [ax for ax in axes if ax not in ("S", "T")]
        same_dims = len(in_shape_tile) == len(restored.shape)

        if same_dims:
            # C dim is that of target_shape
            assert restored.shape[new_axes.index("C")] == target_shape[0]
        else:
            c_removed = len(in_shape_tile) == len(restored.shape) + 1
            set_in_shape = set(in_shape_tile)
            set_restored_shape = set(restored.shape)

            if c_removed:
                # C dim is removed from restored.shape, difference should be original C
                assert set_in_shape - set_restored_shape == {
                    in_shape[new_axes.index("C")]
                }
            else:
                # C dim is added to restored.shape, difference should be target C
                assert set_restored_shape - set_in_shape == {target_shape[0]}

    @pytest.mark.parametrize(
        "in_shape, axes, target_shape",
        _ORDERED_TRANSFORMED
        + _DISORDERED_TRANSFORMED
        + _CHANNEL_MISMATCH
        + _CHANNEL_MISMATCH_DISORDERED,
    )
    def test_stitch_slices_channel_mismatch(self, in_shape, axes, target_shape):
        """Test stitch slices when channel dimensions differ."""
        stitch = 2
        crop = 8
        if "Z" in axes:
            stitch_coords = (stitch,) * 3
            crop_size = (crop,) * 3
        else:
            stitch_coords = (stitch,) * 2
            crop_size = (crop,) * 2

        if "S" in axes or "T" in axes:
            s_idx = 2
        else:
            s_idx = 0

        transform = RestoredAxesTransform(axes, in_shape, target_shape, False)
        stitch_slices = transform.stitch_slices(s_idx, stitch_coords, crop_size)

        for i, ax in enumerate(transform.restored_array_axes):
            if ax in "ZYX":
                assert stitch_slices[i].start == stitch
                assert stitch_slices[i].stop == stitch + crop
            elif ax == "C":
                assert stitch_slices[i].start == 0
                assert stitch_slices[i].stop == transform.current_c_size
            else:  # S, T indexed by an int
                assert isinstance(stitch_slices[i], int)
