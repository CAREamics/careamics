import numpy as np
import pytest

from careamics.transforms import NDFlip


@pytest.mark.parametrize(
    "shape",
    [
        # 2D
        (1, 2, 2),
        (2, 2, 2),
        # 3D
        (1, 2, 2, 2),
        (2, 2, 2, 2),
    ],
)
def test_flip_nd(ordered_array, shape):
    """Test flipping for 2D and 3D arrays."""
    # create array
    array: np.ndarray = ordered_array(shape)

    # create augmentation
    aug = NDFlip(seed=42)
    r = np.random.default_rng(seed=42)

    # potential flips
    axes = [-2, -1]
    flips = [np.flip(array, axis=axis) for axis in axes]

    # apply augmentation 5 times
    for _ in range(4):
        augmented, _ = aug(array)

        assert np.array_equal(augmented, flips[r.choice(axes)])


def test_flip_mask(ordered_array):
    """Test flipping masks in 3D."""
    # create array
    array = ordered_array((4, 2, 2, 2))
    mask = array[2:, ...]
    array = array[:2, ...]

    # create augmentation
    aug = NDFlip(seed=42)
    r = np.random.default_rng(seed=42)

    # potential flips on Y and X axes
    axes = [-2, -1]
    array_flips = [np.flip(array, axis=axis) for axis in axes]
    mask_flips = [np.flip(mask, axis=axis) for axis in axes]

    # apply augmentation 5 times
    for _ in range(5):
        aug_array, aug_mask = aug(patch=array, target=mask)
        axis = r.choice(axes)

        assert np.array_equal(aug_array, array_flips[axis])
        assert np.array_equal(aug_mask, mask_flips[axis])
