import pytest

import numpy as np

from n2v.dataloader_utils.dataloader_utils import (
    compute_patch_steps,
    _compute_number_of_patches,
    compute_overlap,
    compute_reshaped_view,
    are_axes_valid,
    compute_overlap_auto,
    compute_overlap_predict,
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


@pytest.mark.parametrize(
    "patch_sizes, overlaps, expected_window, expected_steps",
    [
        ((5, 5), (2, 2), (5, 5), (3, 3)),
        ((None, 5, 8), (None, 1, 2), (5, 8), (4, 6)),
        ((8, 4, 7), (2, 1, 3), (8, 4, 7), (6, 3, 4)),
    ],
)
def test_compute_view_windows(patch_sizes, overlaps, expected_window, expected_steps):
    window, steps = compute_view_windows(patch_sizes, overlaps)

    assert window == expected_window
    assert steps == expected_steps


@pytest.mark.parametrize("arr", [(67,), (72,), (321,)])
@pytest.mark.parametrize(
    "patch_shape",
    [(32,), (64,), (256,)],
)
@pytest.mark.parametrize("overlap", [(16,), (21,), (24,), (32,)])
def test_compute_overlap_predict(arr, patch_shape, overlap):
    if any(patch_shape[i] <= overlap[i] for i in range(len(patch_shape))) or any(
        patch_shape[i] > arr[i] for i in range(len(patch_shape))
    ):
        pass
    else:
        step, updated_overlap = compute_overlap_predict(
            np.ones((1, *arr)), patch_shape, overlap
        )
        assert (arr[0] - updated_overlap[0]) / (
            patch_shape[0] - updated_overlap[0]
        ).is_integer()


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

    win = (1,) + window_shape
    step = (1,) + steps
    output_shape = (-1,) + window_shape

    # compute views
    output = compute_reshaped_view(array_2D, win, step, output_shape)

    # check the number of patches
    n_patches = [
        np.ceil((array_2D.shape[1 + i] - window_shape[i] + 1) / steps[i]).astype(int)
        for i in range(len(window_shape))
    ]
    assert output.shape == (np.prod(n_patches),) + window_shape

    # check all patches
    for i in range(n_patches[0]):
        for j in range(n_patches[1]):
            start_i = i * steps[0]
            end_i = i * steps[0] + window_shape[0]
            start_j = j * steps[1]
            end_j = j * steps[1] + window_shape[1]

            patch = array_2D[0, start_i:end_i, start_j:end_j]
            assert np.all(output[i * n_patches[1] + j] == patch)


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

    win = (1,) + window_shape
    step = (1,) + steps
    output_shape = (-1,) + window_shape

    # compute views
    output = compute_reshaped_view(array_3D, win, step, output_shape)

    # check the number of patches
    n_patches = [
        np.ceil((array_3D.shape[1 + i] - window_shape[i] + 1) / steps[i]).astype(int)
        for i in range(len(window_shape))
    ]
    assert output.shape == (np.prod(n_patches),) + window_shape

    # check all patches
    for i in range(n_patches[0]):
        for j in range(n_patches[1]):
            for k in range(n_patches[2]):
                start_i = i * steps[0]
                end_i = i * steps[0] + window_shape[0]
                start_j = j * steps[1]
                end_j = j * steps[1] + window_shape[1]
                start_k = k * steps[2]
                end_k = k * steps[2] + window_shape[2]

                patch = array_3D[0, start_i:end_i, start_j:end_j, start_k:end_k]
                assert np.all(
                    output[i * n_patches[1] * n_patches[2] + j * n_patches[2] + k]
                    == patch
                )


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


@pytest.mark.parametrize("d", [i for i in range(100, 128)])
@pytest.mark.parametrize("patch_size", [2**i for i in range(4, 6)])
def test_compute_perfect_overlap(d, patch_size):
    if patch_size >= d:
        pytest.skip("Patch size must be smaller than the dimension")
    else:
        step, overlap = compute_overlap_auto(d, patch_size)

        n = (d - patch_size) / (patch_size - overlap)
        print(step)
        assert n.is_integer()
