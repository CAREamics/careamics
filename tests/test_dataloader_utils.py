import pytest

import numpy as np

from n2v.dataloader_utils.dataloader_utils import (
    compute_view_windows,
    _compute_patch_steps,
    _compute_number_of_patches,
    compute_overlap,
    compute_reshaped_view
)

@pytest.mark.parametrize("shape, patch_sizes, expected",
                         [
                            ((1, 10, 10), (10, 5), (1, 2)),
                            ((1, 9, 9), (4, 3), (3, 3)),
                            ((1, 10, 9), (3, 5), (4, 2)),
                            ((1, 5, 9, 10), (2, 3, 5), (3, 3, 2))
                         ])
def test_compute_number_of_patches(shape, patch_sizes, expected):
    """Test computing number of patches"""
    arr = np.ones(shape)

    assert _compute_number_of_patches(arr, patch_sizes) == expected

@pytest.mark.parametrize("shape, patch_sizes, expected",
                         [
                            ((1, 10, 10), (10, 5), (0, 0)),
                            ((1, 9, 9), (4, 3), (2, 0)),
                            ((1, 10, 9), (3, 5), (1, 1))
                         ])
def test_compute_overlap(shape, patch_sizes, expected):
    """Test computing overlap between patches"""
    arr = np.ones(shape)

    assert compute_overlap(arr, patch_sizes) == expected


@pytest.mark.parametrize("dims", [2, 3])
@pytest.mark.parametrize("patch_size", [2, 3])
@pytest.mark.parametrize("overlap",[0, 1, 4])
def test_compute_patch_steps(dims, patch_size, overlap):
    """Test computing patch steps"""
    patch_sizes = (patch_size,) * dims
    overlaps = (overlap,) * dims
    expected = (min(patch_size - overlap, patch_size),) * dims

    assert _compute_patch_steps(patch_sizes, overlaps) == expected


@pytest.mark.parametrize("patch_sizes, overlaps, expected_window, expected_steps",
                         [
                            ((5, 5), (2, 2), (5, 5), (3, 3)),
                            ((None, 5, 8), (None, 1, 2), (5, 8), (4, 6)),
                            ((8, 4, 7), (2, 1, 3), (8, 4, 7), (6, 3, 4)),
                         ])
def test_compute_view_windows(patch_sizes, overlaps, expected_window, expected_steps):
    window, steps = compute_view_windows(patch_sizes, overlaps)
    
    assert  window == expected_window
    assert steps == expected_steps



@pytest.mark.parametrize("window_shape, steps", 
                         [
                                ((5, 5), (1, 1)),
                                ((5, 5), (2, 3)),
                                ((5, 7), (1, 1)),
                         ])
def test_compute_reshaped_view_2d(array_2D, window_shape, steps):
    """Test computing reshaped view of an array of shape (1, 10, 9)."""
    
    win = (1,) + window_shape
    step = (1,) + steps
    output_shape = (-1,) + window_shape

    # compute views
    output = compute_reshaped_view(array_2D, win, step, output_shape)
    
    # check the number of patches
    n_patches = [
        np.ceil((array_2D.shape[1+i]-window_shape[i]+1)/steps[i]).astype(int)
        for i in range(len(window_shape))
    ]
    assert output.shape == (np.prod(n_patches),) + window_shape

    # check all patches
    for i in range(n_patches[0]):
        for j in range(n_patches[1]):
            start_i = i*steps[0]
            end_i = i*steps[0]+window_shape[0]
            start_j = j*steps[1]
            end_j = j*steps[1]+window_shape[1]

            patch = array_2D[0, start_i:end_i, start_j:end_j]
            assert np.all(output[i*n_patches[1]+j] == patch)
    

@pytest.mark.parametrize("window_shape, steps",
                         [
                                ((1, 5, 5), (2, 1, 2)),
                                ((2, 5, 5), (2, 3, 4)),
                                ((3, 7, 8), (1, 1, 3)),
                         ])
def test_compute_reshaped_view_3d(array_3D, window_shape, steps):
    """Test computing reshaped view of an array of shape (1, 5, 10, 9)."""
    
    win = (1,) + window_shape
    step = (1,) + steps
    output_shape = (-1,) + window_shape

    # compute views
    output = compute_reshaped_view(array_3D, win, step, output_shape)
    
    # check the number of patches
    n_patches = [
        np.ceil((array_3D.shape[1+i]-window_shape[i]+1)/steps[i]).astype(int)
        for i in range(len(window_shape))
    ]
    assert output.shape == (np.prod(n_patches),) + window_shape

    # check all patches
    for i in range(n_patches[0]):
        for j in range(n_patches[1]):
            for k in range(n_patches[2]):
                start_i = i*steps[0]
                end_i = i*steps[0]+window_shape[0]
                start_j = j*steps[1]
                end_j = j*steps[1]+window_shape[1]
                start_k = k*steps[2]
                end_k = k*steps[2]+window_shape[2]

                patch = array_3D[0, start_i:end_i, start_j:end_j, start_k:end_k]
                assert np.all(output[i*n_patches[1]*n_patches[2]+j*n_patches[2]+k] == patch)