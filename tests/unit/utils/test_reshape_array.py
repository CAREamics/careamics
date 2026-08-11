import numpy as np
import pytest

from careamics.utils.reshape_array import (
    AxesTransform,
    RestoredAxesTransform,
    reshape_array,
    restore_array,
)

# --- Test utilities


def _array_to_tile(input_shape, axes, target_shape, target_axes, output_shape):
    """Create the input to restore_tile tests from the restore_array test inputs."""
    tile_output = tuple(
        dim
        for axis, dim in zip(target_axes, output_shape, strict=True)
        if axis not in "ST"
    )
    return input_shape, axes, target_shape[1:], target_axes, tile_output


def _transformed_shape(input_shape, axes, target_axes, output_shape):
    """Get the transformed shape from the restore_array test inputs."""
    transform = AxesTransform(axes, input_shape)
    return input_shape, axes, transform.transformed_shape, target_axes, output_shape


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
    _transformed_shape(in_sh, axes, tar_axes, out_sh)
    for in_sh, axes, tar_axes, out_sh in [
        # shape, axes, target_axes, expected output_shape
        ((XY_S, XY_S), "YX", "XY", (XY_S, XY_S)),
        ((C_S, XY_S, XY_S), "CYX", "YXC", (XY_S, XY_S, C_S)),
        ((S_S, XY_S, XY_S), "SYX", "YXS", (XY_S, XY_S, S_S)),
        ((T_S, XY_S, XY_S), "TYX", "XYT", (XY_S, XY_S, T_S)),
        ((S_S, C_S, XY_S, XY_S), "SCYX", "SYXC", (S_S, XY_S, XY_S, C_S)),
        ((T_S, C_S, XY_S, XY_S), "TCYX", "TYXC", (T_S, XY_S, XY_S, C_S)),
        ((S_S, T_S, C_S, XY_S, XY_S), "STCYX", "TSYXC", (T_S, S_S, XY_S, XY_S, C_S)),
        ((Z_S, XY_S, XY_S), "ZYX", "XZY", (XY_S, Z_S, XY_S)),
        ((C_S, Z_S, XY_S, XY_S), "CZYX", "ZYXC", (Z_S, XY_S, XY_S, C_S)),
        ((S_S, Z_S, XY_S, XY_S), "SZYX", "ZYXS", (Z_S, XY_S, XY_S, S_S)),
        ((T_S, Z_S, XY_S, XY_S), "TZYX", "ZYXT", (Z_S, XY_S, XY_S, T_S)),
        ((S_S, C_S, Z_S, XY_S, XY_S), "SCZYX", "SZYXC", (S_S, Z_S, XY_S, XY_S, C_S)),
        ((T_S, C_S, Z_S, XY_S, XY_S), "TCZYX", "TZYXC", (T_S, Z_S, XY_S, XY_S, C_S)),
        (
            (S_S, T_S, C_S, Z_S, XY_S, XY_S),
            "STCZYX",
            "TSZYXC",
            (T_S, S_S, Z_S, XY_S, XY_S, C_S),
        ),
    ]
]

_DISORDERED_TRANSFORMED = [
    _transformed_shape(in_sh, axes, tar_axes, out_sh)
    for in_sh, axes, tar_axes, out_sh in [
        # shape, axes, target_axes, expected output_shape
        ((XY_S, XY_S, C_S), "YXC", "CXY", (C_S, XY_S, XY_S)),
        ((XY_S, XY_S, Z_S), "YXZ", "XZY", (XY_S, Z_S, XY_S)),
        ((XY_S, XY_S, T_S, Z_S), "YXTZ", "TZYX", (T_S, Z_S, XY_S, XY_S)),
        ((XY_S, XY_S, Z_S, C_S), "YXZC", "CZYX", (C_S, Z_S, XY_S, XY_S)),
        ((XY_S, XY_S, S_S), "YXS", "SYX", (S_S, XY_S, XY_S)),
        (
            (C_S, XY_S, XY_S, T_S, Z_S, S_S),
            "CYXTZS",
            "TSZYXC",
            (T_S, S_S, Z_S, XY_S, XY_S, C_S),
        ),
    ]
]

_ORDERED_TRANSFORMED_TILE = [
    _array_to_tile(in_sh, axes, tar_sh, tar_axes, out_sh)
    for in_sh, axes, tar_sh, tar_axes, out_sh in _ORDERED_TRANSFORMED
]

_DISORDERED_TRANSFORMED_TILE = [
    _array_to_tile(in_sh, axes, tar_sh, tar_axes, out_sh)
    for in_sh, axes, tar_sh, tar_axes, out_sh in _DISORDERED_TRANSFORMED
]

# Input-target with different C dimensions
_CHANNEL_REMOVED_CANONICAL = [
    # shape, axes, target_shape, target_axes, expected output_shape
    ((C_S, XY_S, XY_S), "CYX", (1, 1, XY_S, XY_S), "CYX", (1, XY_S, XY_S)),
    ((C_S, XY_S, XY_S), "CYX", (1, 1, XY_S, XY_S), "YX", (XY_S, XY_S)),
    (
        (S_S, C_S, Z_S, XY_S, XY_S),
        "SCZYX",
        (S_S, 1, Z_S, XY_S, XY_S),
        "SCZYX",
        (S_S, 1, Z_S, XY_S, XY_S),
    ),
    ((T_S, C_S, XY_S, XY_S), "TCYX", (T_S, 1, XY_S, XY_S), "TYX", (T_S, XY_S, XY_S)),
    (
        (T_S, C_S, XY_S, XY_S),
        "TCYX",
        (T_S, 1, XY_S, XY_S),
        "TCYX",
        (T_S, 1, XY_S, XY_S),
    ),
    (
        (S_S, T_S, C_S, XY_S, XY_S),
        "STCYX",
        (S_S * T_S, 1, XY_S, XY_S),
        "STCYX",
        (S_S, T_S, 1, XY_S, XY_S),
    ),
    (
        (S_S, T_S, C_S, XY_S, XY_S),
        "STCYX",
        (S_S * T_S, 1, XY_S, XY_S),
        "STYX",
        (S_S, T_S, XY_S, XY_S),
    ),
]

_CHANNEL_REMOVED_NON_CANONICAL = [
    ((C_S, XY_S, XY_S), "CYX", (1, 1, XY_S, XY_S), "YXC", (XY_S, XY_S, 1)),
    (
        (C_S, Z_S, XY_S, XY_S),
        "CZYX",
        (1, 1, Z_S, XY_S, XY_S),
        "ZYXC",
        (Z_S, XY_S, XY_S, 1),
    ),
    (
        (S_S, C_S, XY_S, XY_S),
        "SCYX",
        (S_S, 1, XY_S, XY_S),
        "SYXC",
        (S_S, XY_S, XY_S, 1),
    ),
    (
        (S_S, C_S, Z_S, XY_S, XY_S),
        "SCZYX",
        (S_S, 1, Z_S, XY_S, XY_S),
        "SZYXC",
        (S_S, Z_S, XY_S, XY_S, 1),
    ),
    (
        (T_S, C_S, XY_S, XY_S),
        "TCYX",
        (T_S, 1, XY_S, XY_S),
        "TYXC",
        (T_S, XY_S, XY_S, 1),
    ),
    (
        (T_S, C_S, Z_S, XY_S, XY_S),
        "TCZYX",
        (T_S, 1, Z_S, XY_S, XY_S),
        "TZYXC",
        (T_S, Z_S, XY_S, XY_S, 1),
    ),
    (
        (S_S, T_S, C_S, XY_S, XY_S),
        "STCYX",
        (S_S * T_S, 1, XY_S, XY_S),
        "TSYXC",
        (T_S, S_S, XY_S, XY_S, 1),
    ),
    (
        (S_S, T_S, C_S, Z_S, XY_S, XY_S),
        "STCZYX",
        (S_S * T_S, 1, Z_S, XY_S, XY_S),
        "TSZYXC",
        (T_S, S_S, Z_S, XY_S, XY_S, 1),
    ),
]

_CHANNEL_REMOVED = _CHANNEL_REMOVED_CANONICAL + _CHANNEL_REMOVED_NON_CANONICAL

_CHANNEL_ADDED = [
    # shape, axes, target_shape, target_axes, expected output_shape
    ((XY_S, XY_S), "YX", (1, C_S, XY_S, XY_S), "YXC", (XY_S, XY_S, C_S)),
    (
        (Z_S, XY_S, XY_S),
        "ZYX",
        (1, C_S, Z_S, XY_S, XY_S),
        "ZYXC",
        (Z_S, XY_S, XY_S, C_S),
    ),
    ((S_S, XY_S, XY_S), "SYX", (S_S, C_S, XY_S, XY_S), "SYXC", (S_S, XY_S, XY_S, C_S)),
    (
        (S_S, Z_S, XY_S, XY_S),
        "SZYX",
        (S_S, C_S, Z_S, XY_S, XY_S),
        "SZYXC",
        (S_S, Z_S, XY_S, XY_S, C_S),
    ),
    ((T_S, XY_S, XY_S), "TYX", (T_S, C_S, XY_S, XY_S), "TYXC", (T_S, XY_S, XY_S, C_S)),
    (
        (T_S, Z_S, XY_S, XY_S),
        "TZYX",
        (T_S, C_S, Z_S, XY_S, XY_S),
        "TZYXC",
        (T_S, Z_S, XY_S, XY_S, C_S),
    ),
    (
        (S_S, T_S, XY_S, XY_S),
        "STYX",
        (S_S * T_S, C_S, XY_S, XY_S),
        "TSYXC",
        (T_S, S_S, XY_S, XY_S, C_S),
    ),
    (
        (S_S, T_S, Z_S, XY_S, XY_S),
        "STZYX",
        (S_S * T_S, C_S, Z_S, XY_S, XY_S),
        "TSZYXC",
        (T_S, S_S, Z_S, XY_S, XY_S, C_S),
    ),
]

_CHANNEL_CHANGED = [
    # shape, axes, target_shape, target_axes, expected output_shape
    ((C_S, XY_S, XY_S), "CYX", (1, OUT_C_S, XY_S, XY_S), "YXC", (XY_S, XY_S, OUT_C_S)),
    (
        (C_S, Z_S, XY_S, XY_S),
        "CZYX",
        (1, OUT_C_S, Z_S, XY_S, XY_S),
        "ZYXC",
        (Z_S, XY_S, XY_S, OUT_C_S),
    ),
    (
        (S_S, C_S, XY_S, XY_S),
        "SCYX",
        (S_S, OUT_C_S, XY_S, XY_S),
        "SYXC",
        (S_S, XY_S, XY_S, OUT_C_S),
    ),
    (
        (S_S, C_S, Z_S, XY_S, XY_S),
        "SCZYX",
        (S_S, OUT_C_S, Z_S, XY_S, XY_S),
        "SZYXC",
        (S_S, Z_S, XY_S, XY_S, OUT_C_S),
    ),
    (
        (T_S, C_S, XY_S, XY_S),
        "TCYX",
        (T_S, OUT_C_S, XY_S, XY_S),
        "TYXC",
        (T_S, XY_S, XY_S, OUT_C_S),
    ),
    (
        (S_S, T_S, C_S, XY_S, XY_S),
        "STCYX",
        (S_S * T_S, OUT_C_S, XY_S, XY_S),
        "TSYXC",
        (T_S, S_S, XY_S, XY_S, OUT_C_S),
    ),
    ((1, XY_S, XY_S), "CYX", (1, OUT_C_S, XY_S, XY_S), "YXC", (XY_S, XY_S, OUT_C_S)),
    (
        (S_S, 1, XY_S, XY_S),
        "SCYX",
        (S_S, OUT_C_S, XY_S, XY_S),
        "SYXC",
        (S_S, XY_S, XY_S, OUT_C_S),
    ),
]

_CHANNEL_MISMATCH = _CHANNEL_REMOVED + _CHANNEL_ADDED + _CHANNEL_CHANGED

_CHANNEL_REMOVED_DISORDERED = [
    # shape, axes, target_shape, target_axes, expected output_shape
    ((XY_S, XY_S, C_S), "YXC", (1, 1, XY_S, XY_S), "YXC", (XY_S, XY_S, 1)),
    (
        (XY_S, XY_S, C_S, Z_S),
        "YXCZ",
        (1, 1, Z_S, XY_S, XY_S),
        "YXZC",
        (XY_S, XY_S, Z_S, 1),
    ),
    (
        (XY_S, XY_S, C_S, T_S),
        "YXCT",
        (T_S, 1, XY_S, XY_S),
        "TYXC",
        (T_S, XY_S, XY_S, 1),
    ),
    (
        (XY_S, XY_S, T_S, C_S, Z_S),
        "YXTCZ",
        (T_S, 1, Z_S, XY_S, XY_S),
        "TYXZC",
        (T_S, XY_S, XY_S, Z_S, 1),
    ),
    (
        (XY_S, XY_S, T_S, C_S, S_S, Z_S),
        "YXTCSZ",
        (S_S * T_S, 1, Z_S, XY_S, XY_S),
        "TSYXZC",
        (T_S, S_S, XY_S, XY_S, Z_S, 1),
    ),
]

_CHANNEL_ADDED_DISORDERED = [
    # shape, axes, target_shape, target_axes, expected output_shape
    (
        (XY_S, XY_S, Z_S),
        "YXZ",
        (1, C_S, Z_S, XY_S, XY_S),
        "YXCZ",
        (XY_S, XY_S, C_S, Z_S),
    ),
    ((XY_S, XY_S, T_S), "YXT", (T_S, C_S, XY_S, XY_S), "TYXC", (T_S, XY_S, XY_S, C_S)),
    (
        (XY_S, XY_S, T_S, Z_S),
        "YXTZ",
        (T_S, C_S, Z_S, XY_S, XY_S),
        "TYXZC",
        (T_S, XY_S, XY_S, Z_S, C_S),
    ),
    (
        (XY_S, XY_S, T_S, S_S, Z_S),
        "YXTSZ",
        (S_S * T_S, C_S, Z_S, XY_S, XY_S),
        "TSYXZC",
        (T_S, S_S, XY_S, XY_S, Z_S, C_S),
    ),
]


_CHANNEL_CHANGED_DISORDERED = [
    # shape, axes, target_shape, target_axes, expected output_shape
    ((XY_S, XY_S, C_S), "YXC", (1, OUT_C_S, XY_S, XY_S), "YXC", (XY_S, XY_S, OUT_C_S)),
    (
        (XY_S, XY_S, C_S, Z_S),
        "YXCZ",
        (1, OUT_C_S, Z_S, XY_S, XY_S),
        "YXZC",
        (XY_S, XY_S, Z_S, OUT_C_S),
    ),
    (
        (C_S, XY_S, XY_S, T_S, S_S),
        "CYXTS",
        (S_S * T_S, OUT_C_S, XY_S, XY_S),
        "TSYXC",
        (T_S, S_S, XY_S, XY_S, OUT_C_S),
    ),
    (
        (C_S, XY_S, XY_S, T_S, Z_S, S_S),
        "CYXTZS",
        (S_S * T_S, OUT_C_S, Z_S, XY_S, XY_S),
        "TSYXZC",
        (T_S, S_S, XY_S, XY_S, Z_S, OUT_C_S),
    ),
    ((XY_S, XY_S, 1), "YXC", (1, OUT_C_S, XY_S, XY_S), "YXC", (XY_S, XY_S, OUT_C_S)),
]

_CHANNEL_MISMATCH_DISORDERED = (
    _CHANNEL_REMOVED_DISORDERED
    + _CHANNEL_ADDED_DISORDERED
    + _CHANNEL_CHANGED_DISORDERED
)


_CHANNEL_MISMATCH_TILE = [
    _array_to_tile(in_sh, axes, tar_sh, tar_axes, out_sh)
    for in_sh, axes, tar_sh, tar_axes, out_sh in _CHANNEL_MISMATCH
]

_CHANNEL_MISMATCH_TILE_DISORDERED = [
    _array_to_tile(in_sh, axes, tar_sh, tar_axes, out_sh)
    for in_sh, axes, tar_sh, tar_axes, out_sh in _CHANNEL_MISMATCH_DISORDERED
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


def test_restore_array_wrong_ndim():
    with pytest.raises(ValueError, match="Expected 4D"):
        restore_array(np.zeros((32, 32, 32)), "YX", (32, 32))


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
            RestoredAxesTransform(axes, shape, axes, target_shape, is_tile)

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
            RestoredAxesTransform(axes, shape, axes, target_shape, is_tile)

    @pytest.mark.parametrize(
        "in_shape, axes, target_shape, target_axes, output_shape, canonical",
        [t + (True,) for t in _CHANNEL_REMOVED_CANONICAL]
        + [t + (False,) for t in _CHANNEL_REMOVED_NON_CANONICAL],
    )
    def test_canonical_order(
        self, in_shape, axes, target_shape, target_axes, output_shape, canonical
    ):
        """Test that the canonical order is always SCZYX."""
        transform = RestoredAxesTransform(
            axes, in_shape, target_axes, target_shape, current_is_tile=False
        )
        assert transform.canonical_order == canonical

    @pytest.mark.parametrize(
        "in_shape, axes, target_shape, target_axes, output_shape",
        _ORDERED_TRANSFORMED + _DISORDERED_TRANSFORMED,
    )
    def test_restored_array_shape(
        self, in_shape, axes, target_shape, target_axes, output_shape
    ):
        """Test that the restored array shape is the target shape in the absence of
        channel mismatch."""
        transform = RestoredAxesTransform(
            axes, in_shape, target_axes, target_shape, current_is_tile=False
        )
        # test that the restored array shape is the output shape
        assert transform.restored_array_shape == output_shape

    @pytest.mark.parametrize(
        "in_shape, axes, target_shape, target_axes, output_shape",
        _ORDERED_TRANSFORMED
        + _DISORDERED_TRANSFORMED
        + _CHANNEL_CHANGED
        + _CHANNEL_CHANGED_DISORDERED,
    )
    def test_restored_axes(
        self, in_shape, axes, target_shape, target_axes, output_shape
    ):
        """Test that the restored array axes are the original axes in the absence of
        channel mismatch."""
        transform = RestoredAxesTransform(
            axes, in_shape, target_axes, target_shape, current_is_tile=False
        )
        # test that the restored array axes are the original axes
        assert "".join(transform.restored_axes) == target_axes

    @pytest.mark.parametrize(
        "in_shape, axes, target_shape, target_axes, output_shape",
        _ORDERED_TRANSFORMED
        + _DISORDERED_TRANSFORMED
        + _CHANNEL_MISMATCH
        + _CHANNEL_MISMATCH_DISORDERED,
    )
    def test_current_c_size(
        self, in_shape, axes, target_shape, target_axes, output_shape
    ):
        """Test that the current channel is correct."""
        transform = RestoredAxesTransform(
            axes, in_shape, target_axes, target_shape, current_is_tile=False
        )
        assert transform.current_c_size == target_shape[1]

    # TODO
    @pytest.mark.parametrize(
        "in_shape, axes, target_shape, target_axes, output_shape",
        _ORDERED_TRANSFORMED
        + _DISORDERED_TRANSFORMED
        + _CHANNEL_MISMATCH
        + _CHANNEL_MISMATCH_DISORDERED,
    )
    def test_restore_array(
        self, in_shape, axes, target_shape, target_axes, output_shape
    ):
        """Test that the restored array shape is the original shape in the absence of
        channel mismatch."""
        transform = RestoredAxesTransform(
            axes, in_shape, target_axes, target_shape, current_is_tile=False
        )
        restored = transform.restore(np.zeros(target_shape))

        # test that the restored array shape is the original shape
        assert restored.shape == output_shape

    @pytest.mark.parametrize(
        "in_shape, axes, target_shape, target_axes, output_shape",
        _ORDERED_TRANSFORMED_TILE
        + _DISORDERED_TRANSFORMED_TILE
        + _CHANNEL_MISMATCH_TILE
        + _CHANNEL_MISMATCH_TILE_DISORDERED,
    )
    def test_restore_tile(
        self, in_shape, axes, target_shape, target_axes, output_shape
    ):
        """Test that the restored array shape is the original shape in the absence of
        channel mismatch."""
        transform = RestoredAxesTransform(
            axes, in_shape, target_axes, target_shape, current_is_tile=True
        )
        restored = transform.restore(np.zeros(target_shape))

        assert restored.shape == output_shape

    @pytest.mark.parametrize(
        "in_shape, axes, target_shape, target_axes, output_shape",
        _ORDERED_TRANSFORMED
        + _DISORDERED_TRANSFORMED
        + _CHANNEL_MISMATCH
        + _CHANNEL_MISMATCH_DISORDERED,
    )
    def test_stitch_slices_channel_mismatch(
        self, in_shape, axes, target_shape, target_axes, output_shape
    ):
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

        transform = RestoredAxesTransform(
            axes, in_shape, target_axes, target_shape, current_is_tile=False
        )
        stitch_slices = transform.stitch_slices(s_idx, stitch_coords, crop_size)

        for i, ax in enumerate(target_axes):
            if ax in "ZYX":
                assert stitch_slices[i].start == stitch
                assert stitch_slices[i].stop == stitch + crop
            elif ax == "C":
                assert stitch_slices[i].start == 0
                assert stitch_slices[i].stop == transform.current_c_size
            else:  # S, T indexed by an int
                assert isinstance(stitch_slices[i], int)

    @pytest.mark.parametrize(
        "in_shape, axes, target_axes, shape, expected_shape",
        [
            ((32, 32), "YX", "YX", (8, 8), (8, 8)),
            ((32, 32), "YX", "YXC", (8, 8), (8, 8, 1)),
            ((32, 32), "YX", "CYX", (8, 8), (1, 8, 8)),
            ((3, 32, 32), "CYX", "CYX", (5, 8, 8), (5, 8, 8)),
            ((3, 32, 32), "CYX", "YXC", (5, 8, 8), (8, 8, 5)),
            ((3, 32, 32), "CYX", "YX", (5, 8, 8), (8, 8)),
        ],
    )
    def test_adjust_shape(self, in_shape, axes, target_axes, shape, expected_shape):
        """Test that restoring shapes remove or add "C" dimension."""
        transform = RestoredAxesTransform(
            axes,
            in_shape,
            target_axes,
            AxesTransform(axes, in_shape).transformed_shape,
            False,
        )
        assert transform.adjust_shape(shape) == expected_shape
