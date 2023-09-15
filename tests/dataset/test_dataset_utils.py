import numpy as np
import pytest

from careamics_restoration.dataset.tiling import (
    _compute_number_of_patches,
    compute_crop_and_stitch_coords_1d,
    compute_overlap,
    compute_patch_steps,
    compute_reshaped_view,
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

    win = (1, *window_shape)
    step = (1, *steps)
    output_shape = (-1, *window_shape)

    # compute views
    output = compute_reshaped_view(array, win, step, output_shape)

    # check the number of patches
    n_patches = [
        np.ceil((array.shape[1 + i] - window_shape[i] + 1) / steps[i]).astype(int)
        for i in range(len(window_shape))
    ]
    assert output.shape == (np.prod(n_patches), *window_shape)


@pytest.mark.parametrize("axis_size", [32, 35, 40])
@pytest.mark.parametrize("patch_size, overlap", [(16, 4), (8, 6), (16, 8), (32, 24)])
def test_compute_crop_and_stitch_coords_1d(axis_size, patch_size, overlap):
    crop_coords, stitch_coords, overlap_crop_coords = compute_crop_and_stitch_coords_1d(
        axis_size, patch_size, overlap
    )

    # check that the number of patches is sufficient to cover the whole axis and that
    # the number of coordinates is
    # the same for all three coordinate groups
    num_patches = np.ceil((axis_size - overlap) / (patch_size - overlap)).astype(int)
    assert (
        len(crop_coords)
        == len(stitch_coords)
        == len(overlap_crop_coords)
        == num_patches
    )
    # check if 0 is the first coordinate, axis_size is last coordinate in all three
    # coordinate groups
    assert all(
        all((group[0][0] == 0, group[-1][1] == axis_size))
        for group in [crop_coords, stitch_coords]
    )
    # check if neighboring stitch coordinates are equal
    assert all(
        stitch_coords[i][1] == stitch_coords[i + 1][0]
        for i in range(len(stitch_coords) - 1)
    )

    # check that the crop coordinates cover the whole axis
    assert (
        np.sum(np.array(crop_coords)[:, 1] - np.array(crop_coords)[:, 0])
        == patch_size * num_patches
    )

    # check that the overlap crop coordinates cover the whole axis
    assert (
        np.sum(
            np.array(overlap_crop_coords)[:, 1] - np.array(overlap_crop_coords)[:, 0]
        )
        == axis_size
    )

    # check that shape of all cropped tiles is equal
    assert np.array_equal(
        np.array(overlap_crop_coords)[:, 1] - np.array(overlap_crop_coords)[:, 0],
        np.array(stitch_coords)[:, 1] - np.array(stitch_coords)[:, 0],
    )


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
