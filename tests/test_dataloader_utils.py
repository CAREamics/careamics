import pytest

import numpy as np

from careamics_restoration.dataloader_utils.dataloader_utils import (
    compute_patch_steps,
    _compute_number_of_patches,
    compute_overlap,
    compute_reshaped_view,
    are_axes_valid,
)


@pytest.mark.parametrize(
    "shape, patch_sizes, expected",
    [
        ((1, 10, 10), (10, 5), (1, 2)),
        ((1, 9, 9), (4, 3), (3, 3)),
        ((1, 10, 9), (3, 5), (4, 2)),
        ((1, 5, 9, 10), (2, 3, 5), (3, 3, 2)),
    ],
)
def test_compute_number_of_patches(shape, patch_sizes, expected):
    """Test computing number of patches"""
    arr = np.ones(shape)

    assert _compute_number_of_patches(arr, patch_sizes) == expected


@pytest.mark.parametrize(
    "shape, patch_sizes, expected",
    [
        ((1, 10, 10), (10, 5), (0, 0)),
        ((1, 9, 9), (4, 3), (2, 0)),
        ((1, 10, 9), (3, 5), (1, 1)),
    ],
)
def test_compute_overlap(shape, patch_sizes, expected):
    """Test computing overlap between patches"""
    arr = np.ones(shape)

    assert compute_overlap(arr, patch_sizes) == expected


@pytest.mark.parametrize("dims", [2, 3])
@pytest.mark.parametrize("patch_size", [2, 3])
@pytest.mark.parametrize("overlap", [0, 1, 4])
def test_compute_patch_steps(dims, patch_size, overlap):
    """Test computing patch steps"""
    patch_sizes = (patch_size,) * dims
    overlaps = (overlap,) * dims
    expected = (min(patch_size - overlap, patch_size),) * dims

    assert compute_patch_steps(patch_sizes, overlaps) == expected


def check_compute_reshaped_view(array, window_shape, steps):
    """Check the number of patches"""

    win = (1,) + window_shape
    step = (1,) + steps
    output_shape = (-1,) + window_shape

    # compute views
    output = compute_reshaped_view(array, win, step, output_shape)

    # check the number of patches
    n_patches = [
        np.ceil((array.shape[1 + i] - window_shape[i] + 1) / steps[i]).astype(int)
        for i in range(len(window_shape))
    ]
    assert output.shape == (np.prod(n_patches),) + window_shape


@pytest.mark.parametrize(
    "window_shape, steps",
    [
        ((5, 5), (1, 1)),
        ((5, 5), (2, 3)),
        ((5, 7), (1, 1)),
    ],
)
def test_compute_reshaped_view_2d(array_2D, window_shape, steps):
    """Test computing reshaped view of an array of shape (1, 10, 9)."""
    check_compute_reshaped_view(array_2D, window_shape, steps)


@pytest.mark.parametrize(
    "window_shape, steps",
    [
        ((1, 5, 5), (2, 1, 2)),
        ((2, 5, 5), (2, 3, 4)),
        ((3, 7, 8), (1, 1, 3)),
    ],
)
def test_compute_reshaped_view_3d(array_3D, window_shape, steps):
    """Test computing reshaped view of an array of shape (1, 5, 10, 9)."""
    check_compute_reshaped_view(array_3D, window_shape, steps)


@pytest.mark.parametrize(
    "axes, valid",
    [
        # Passing
        ("yx", True),
        ("Yx", True),
        ("Zyx", True),
        ("TzYX", True),
        ("SZYX", True),
        # Failing due to order
        ("XY", False),
        ("YXZ", False),
        ("YXT", False),
        ("ZTYX", False),
        # too few axes
        ("", False),
        ("X", False),
        # too many axes
        ("STZYX", False),
        # no yx axes
        ("ZT", False),
        ("ZY", False),
        # unsupported axes or axes pair
        ("STYX", False),
        ("CYX", False),
        # repeating characters
        ("YYX", False),
        ("YXY", False),
        # invalid characters
        ("YXm", False),
        ("1YX", False),
    ],
)
def test_are_axes_valid(axes, valid):
    """Test if axes are valid"""
    if valid:
        are_axes_valid(axes)
    else:
        with pytest.raises((ValueError, NotImplementedError)):
            are_axes_valid(axes)
