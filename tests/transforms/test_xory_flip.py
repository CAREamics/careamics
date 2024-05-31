import numpy as np
import pytest

from careamics.transforms import XorYFlip


@pytest.mark.parametrize(
    "shape, axis",
    [
        # 2D
        ((1, 2, 2), -1),
        ((2, 2, 2), -1),
        ((1, 2, 2), -2),
        ((2, 2, 2), -2),
        # 3D
        ((1, 2, 2, 2), -1),
        ((2, 2, 2, 2), -1),
        ((1, 2, 2, 2), -2),
        ((2, 2, 2, 2), -2),
    ],
)
def test_flip_xy(ordered_array, shape, axis):
    """Test flipping for 2D and 3D arrays."""
    # create array
    array: np.ndarray = ordered_array(shape)

    # create augmentation
    aug = XorYFlip(axis=axis, p=1, seed=42)
    r = np.random.default_rng(seed=42)

    # potential flips
    flip = np.flip(array, axis=axis)

    # apply augmentation 5 times
    for _ in range(4):
        r.random()  # consume random number
        augmented, _ = aug(array)

        assert np.array_equal(augmented, flip)


@pytest.mark.parametrize("axis", [-2, -1])
def test_flip_mask(ordered_array, axis):
    """Test flipping masks in 3D."""
    # create array
    array = ordered_array((4, 2, 2, 2))
    mask = array[2:, ...]
    array = array[:2, ...]

    # create augmentation
    aug = XorYFlip(axis=axis, p=1, seed=42)
    r = np.random.default_rng(seed=42)

    # potential flips on Y and X axes
    array_flip = np.flip(array, axis=axis)
    mask_flip = np.flip(mask, axis=axis)

    # apply augmentation 5 times
    for _ in range(5):
        aug_array, aug_mask = aug(patch=array, target=mask)
        r.random()  # consume random number

        assert np.array_equal(aug_array, array_flip)
        assert np.array_equal(aug_mask, mask_flip)


def test_p(ordered_array):
    """Test that the probability of flipping is respected."""
    # create array
    array = ordered_array((2, 2, 2))

    # create augmentation that never applies
    aug = XorYFlip(axis=-1, p=0.0, seed=42)

    # apply augmentation 5 times
    for _ in range(5):
        augmented, _ = aug(array)

        assert np.array_equal(augmented, array)

    # create augmentation that always applies
    aug = XorYFlip(axis=-1, p=1.0, seed=42)

    # apply augmentation 5 times
    for _ in range(5):
        augmented, _ = aug(array)

        assert not np.array_equal(augmented, array)
