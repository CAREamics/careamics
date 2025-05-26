import numpy as np
import pytest

from careamics.dataset_ng.patching_strategies.random_patching import (
    RandomPatchingStrategy,
    _calc_n_patches,
    _generate_random_coords,
)


# Note: get_patch_spec for RandomPatchingStrategy and FixedRandomStrategy is already
#   tested in test_all_strategies
@pytest.mark.parametrize(
    "data_shapes,patch_size,expected_patches",
    [
        [[(2, 1, 32, 32), (1, 1, 19, 37), (3, 1, 14, 9)], (8, 8), 59],
        [[(2, 1, 32, 32), (1, 1, 19, 37), (3, 1, 14, 9)], (8, 5), 92],
        [[(2, 1, 32, 32, 32), (1, 1, 19, 37, 23), (3, 1, 14, 9, 12)], (8, 8, 8), 197],
        [[(2, 1, 32, 32, 32), (1, 1, 19, 37, 23), (3, 1, 14, 9, 12)], (8, 5, 7), 400],
    ],
)
def test_calc_patch_bins(data_shapes, patch_size, expected_patches):
    """Test bins are created as expected"""
    image_stack_index_bins, sample_index_bins, sample_bins = (
        RandomPatchingStrategy._calc_bins(data_shapes, patch_size)
    )
    assert image_stack_index_bins[-1] == sample_index_bins[-1] == expected_patches

    # create image_stack_bins from sample_index_bins and sample_bins
    # The idea to find the bin boundaries in sample_index_bins
    #   that are aligned to the image_stack_bins
    new_image_stack_bins = []
    for bin_boundary in sample_bins:
        idx = bin_boundary - 1
        new_image_stack_bins.append(sample_index_bins[idx])

    assert (np.array(image_stack_index_bins) == new_image_stack_bins).all()


@pytest.mark.parametrize(
    "data_shape,patch_size,expected_patches",
    [
        [(1, 1, 19, 37), (8, 8), 15],
        [(1, 1, 19, 37), (8, 5), 24],
        [(1, 1, 19, 37, 23), (8, 8, 8), 45],
        [(1, 1, 19, 37, 23), (8, 5, 7), 96],
    ],
)
def test_n_patches(data_shape, patch_size, expected_patches):
    spatial_shape = data_shape[2:]
    n_patches = _calc_n_patches(spatial_shape, patch_size)

    assert n_patches == expected_patches


def test_n_patches_raises():
    spatial_shape = (8, 8)
    patch_size = (2, 2, 2)
    with pytest.raises(ValueError):
        _calc_n_patches(spatial_shape, patch_size)


@pytest.mark.parametrize(
    "data_shape,patch_size,iterations",
    [
        [(1, 1, 19, 37), (8, 8), 11],
        [(1, 1, 19, 37), (8, 5), 18],
        [(1, 1, 19, 37, 23), (8, 8, 8), 32],
        [(1, 1, 19, 37, 23), (8, 5, 7), 58],
    ],
)
def test_random_coords(data_shape, patch_size, iterations):
    spatial_shape = data_shape[2:]
    rng = np.random.default_rng(42)
    for _ in range(iterations):
        coords = np.array(_generate_random_coords(spatial_shape, patch_size, rng))
        # validate patch is within spatial bounds
        assert (0 <= coords).all()
        assert (coords + patch_size < np.array(spatial_shape)).all()


def test_random_coords_raises():
    spatial_shape = (8, 8)
    patch_size = (2, 2, 2)
    rng = np.random.default_rng(42)
    with pytest.raises(ValueError):
        _generate_random_coords(spatial_shape, patch_size, rng)
