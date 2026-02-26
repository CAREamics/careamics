import numpy as np
import pytest

from careamics.lightning.dataset_ng.prediction.reshape_arrays import (
    AxesTransform,
    get_original_stitch_slices,
    reshape_array,
    restore_array,
    restore_tile,
)

XY_S = 32
Z_S = 16
C_S = 3
T_S = 4
S_S = 5

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
_UNORDERED_CASES = [
    ((XY_S, XY_S, C_S), "YXC"),
    ((XY_S, XY_S, Z_S), "YXZ"),
    ((XY_S, XY_S, T_S, Z_S), "YXTZ"),
    ((XY_S, XY_S, Z_S, C_S), "YXZC"),
    ((XY_S, XY_S, S_S), "YXS"),
    ((C_S, XY_S, XY_S, T_S, Z_S, S_S), "CYXTZS"),
]


class TestAxesTransform:
    def test_s_added(self):
        t = AxesTransform("YX", (XY_S, XY_S))
        assert t.s_added is True
        assert t.c_added is True
        assert t.t_becomes_s is False
        assert t.st_merged is False

    def test_t_becomes_s(self):
        t = AxesTransform("TYX", (T_S, XY_S, XY_S))
        assert t.t_becomes_s is True
        assert t.s_added is False
        assert t.st_merged is False

    def test_st_merged(self):
        t = AxesTransform("STYX", (S_S, T_S, XY_S, XY_S))
        assert t.st_merged is True
        assert t.s_added is False
        assert t.c_added is True
        assert t.t_becomes_s is False

    def test_c_added(self):
        t = AxesTransform("SYX", (S_S, XY_S, XY_S))
        assert t.c_added is True

    def test_c_not_added(self):
        t = AxesTransform("SCYX", (S_S, C_S, XY_S, XY_S))
        assert t.c_added is False

    def test_has_z(self):
        assert AxesTransform("ZYX", (Z_S, XY_S, XY_S)).has_z is True
        assert AxesTransform("YX", (XY_S, XY_S)).has_z is False

    def test_dl_axes_2d(self):
        assert AxesTransform("YX", (XY_S, XY_S)).dl_axes == "SCYX"

    def test_dl_axes_3d(self):
        assert AxesTransform("ZYX", (Z_S, XY_S, XY_S)).dl_axes == "SCZYX"

    def test_dl_shape_yx(self):
        assert AxesTransform("YX", (XY_S, XY_S)).dl_shape == (1, 1, XY_S, XY_S)

    def test_dl_shape_with_c(self):
        assert AxesTransform("YXC", (XY_S, XY_S, C_S)).dl_shape == (1, C_S, XY_S, XY_S)

    def test_dl_shape_with_st(self):
        transform = AxesTransform("STCYX", (S_S, T_S, C_S, XY_S, XY_S))
        assert transform.dl_shape == (S_S * T_S, C_S, XY_S, XY_S)

    def test_dl_shape_t_as_s(self):
        transform = AxesTransform("TYX", (T_S, XY_S, XY_S))
        assert transform.dl_shape == (T_S, 1, XY_S, XY_S)

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

    @pytest.mark.parametrize("shape, axes", _ORDERED_CASES + _UNORDERED_CASES)
    def test_reshape_array_dl_axes(self, shape, axes):
        """Result should always have S, C, and optionally Z, then Y, X."""
        transform = AxesTransform(axes, shape)
        expected_axes = "SCZYX" if "Z" in axes else "SCYX"
        assert transform.dl_axes == expected_axes


class TestReshapeShape:
    """Test that reshape produces the correct output shape."""

    @pytest.mark.parametrize("shape, axes", _ORDERED_CASES + _UNORDERED_CASES)
    def test_reshape_array_produces_correct_shape(self, shape, axes):
        array = np.zeros(shape)
        result = reshape_array(array, axes, shape)
        transform = AxesTransform(axes, shape)

        assert result.shape == transform.dl_shape
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
        result = reshape_array(array, axes, shape)
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
        result = reshape_array(array, axes, shape)
        assert result.shape[1] == expected_c


class TestRestoreIntegrity:
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


class TestReshapeIntegrity:
    """Test that reshape_array specifically put the values in the right dimensions."""

    def test_identity(self) -> None:
        """Test that `reshape_array` returns the same array if already in
        original shape."""
        original_axes = "SCYX"
        original_shape = (S_S, C_S, XY_S, XY_S)

        array = np.arange(np.prod(original_shape)).reshape(original_shape)
        restored = reshape_array(array, original_axes, original_shape)
        assert np.array_equal(restored, array)

    def test_singleton_s(self) -> None:
        """Test that `reshape_array` adds singleton S axis."""
        original_axes = "CYX"
        original_shape = (C_S, XY_S, XY_S)

        array = np.arange(np.prod(original_shape)).reshape(original_shape)
        restored = reshape_array(array, original_axes, original_shape)
        assert np.array_equal(restored[0], array)

    def test_singleton_c(self) -> None:
        """Test that `reshape_array` adds singleton C axis."""
        original_axes = "SYX"
        original_shape = (S_S, XY_S, XY_S)

        array = np.arange(np.prod(original_shape)).reshape(original_shape)
        restored = reshape_array(array, original_axes, original_shape)
        assert np.array_equal(restored[:, 0, ...], array)

    def test_unflatten_s_and_t(self) -> None:
        """Test that `reshape_array` merges S and T into S."""
        original_axes = "STYX"
        original_shape = (S_S, T_S, XY_S, XY_S)

        array = np.arange(np.prod(original_shape)).reshape(original_shape)
        restored = reshape_array(array, original_axes, original_shape)

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
        restored = reshape_array(array, original_axes, original_shape)
        np.testing.assert_array_equal(restored[:, 0, ...], array)

    def test_reorder_axes(self) -> None:
        """Test that `reshape_array` reorders axes to match original."""
        original_axes = "CYXZS"
        original_shape = (C_S, XY_S, XY_S, Z_S, S_S)

        array = np.arange(np.prod(original_shape)).reshape(original_shape)
        restored = reshape_array(array, original_axes, original_shape)

        for c in range(original_shape[0]):
            for z in range(original_shape[3]):
                for s in range(original_shape[4]):
                    np.testing.assert_array_equal(
                        restored[s, c, z], array[c, :, :, z, s]
                    )


@pytest.mark.parametrize("shape, axes", _ORDERED_CASES + _UNORDERED_CASES)
def test_restore_array_roundtrip(shape, axes):
    """reshape then restore should recover the original array."""
    array = np.arange(np.prod(shape)).reshape(shape)
    reshaped = reshape_array(array, axes, shape)
    restored = restore_array(reshaped, axes, shape)

    assert restored.shape == shape
    np.testing.assert_array_equal(restored, array)


def test_restore_array_wrong_ndim():
    with pytest.raises(ValueError, match="Expected 4D"):
        restore_array(np.zeros((32, 32, 32)), "YX", (32, 32))


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


class TestOriginalStitchSlices:
    @pytest.mark.parametrize("shape, axes", _ORDERED_CASES + _UNORDERED_CASES)
    def test_indexing_on_restored(self, shape, axes):

        if "Z" in axes:
            crop_size = (8, 16, 16)
            stitch_coords = (4, 12, 12)
        else:
            crop_size = (16, 16)
            stitch_coords = (12, 12)

        if "S" in axes or "T" in axes:
            sample_idx = 2
        else:
            sample_idx = 0

        array = np.arange(np.prod(shape)).reshape(shape)
        reshaped = reshape_array(array, axes, shape)
        restored = restore_array(reshaped, axes, shape)

        # get slices and index into restored array
        slices = get_original_stitch_slices(
            reshaped.shape, axes, shape, sample_idx, stitch_coords, crop_size
        )

        crop_axes = [a for a in axes if a in "CZYX"]
        crop_shape = []
        for ax in crop_axes:
            if ax == "C":
                crop_shape.append(shape[axes.index("C")])
            elif ax == "Z":
                crop_shape.append(crop_size[0])
            elif ax == "Y" or ax == "X":
                crop_shape.append(crop_size[-1])

        restored[slices] = np.ones(crop_shape)
