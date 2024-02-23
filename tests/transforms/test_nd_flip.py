import pytest

import numpy as np

from careamics.transforms import NDFlip


def test_randomness(ordered_array):
    """Test randomness of the flipping using the `p` parameter."""
    # create array
    array = ordered_array((2, 2))

    # create augmentation that never applies
    aug = NDFlip(p=0.)

    # apply augmentation
    augmented = aug(image=array)["image"]
    assert np.array_equal(augmented, array)

    # create augmentation that always applies
    aug = NDFlip(p=1.)

    # apply augmentation
    augmented = aug(image=array)["image"]
    assert not np.array_equal(augmented, array)


@pytest.mark.parametrize("shape", [
    # 2D
    (2, 2, 1),
    (2, 2, 2),

    # 3D
    (2, 2, 2, 1),
    (2, 2, 2, 2),
])
def test_flip_nd(ordered_array, shape):
    """Test flipping for 2D and 3D arrays."""
    np.random.seed(42)

    # create array
    array: np.ndarray = ordered_array(shape)

    # create augmentation
    is_3D = len(shape) == 4
    aug = NDFlip(p=1, is_3D=is_3D, flip_z=True)

    # potential flips
    axes = [0, 1, 2] if is_3D else [0, 1]
    flips = [np.flip(array, axis=axis) for axis in axes]

    # apply augmentation 10 times
    augs = []
    for _ in range(10):
        augmented = aug(image=array)["image"]

        # check that the augmented array is one of the potential flips
        which_axes = [np.array_equal(augmented, flip) for flip in flips]
        
        assert any(which_axes)
        augs.append(which_axes.index(True))

    # check that all flips were applied
    assert set(augs) == set(axes)


def test_flip_z(ordered_array):
    """Test turning the Z flipping off."""
    np.random.seed(42)

    # create array
    array: np.ndarray = ordered_array((2, 2, 2, 2))

    # create augmentation
    aug = NDFlip(p=1, is_3D=True, flip_z=False)

    # potential flips on Y and X axes
    flips = [np.flip(array, axis=1), np.flip(array, axis=2)]

    # apply augmentation 10 times
    augs = []
    for _ in range(10):
        augmented = aug(image=array)["image"]

        # check that the augmented array is one of the potential flips
        which_axes = [np.array_equal(augmented, flip) for flip in flips]

        assert any(which_axes)
        augs.append(which_axes.index(True))

    # check that all flips were applied (first and second flip)
    assert set(augs) == {0, 1}


def test_flip_mask(ordered_array):
    """Test flipping masks in 3D."""
    np.random.seed(42)

    # create array
    array: np.ndarray = ordered_array((2, 2, 2, 4))
    mask = array[..., 2:]
    array = array[..., :2]

    # create augmentation
    aug = NDFlip(p=1, is_3D=True, flip_z=True)

    # potential flips on Y and X axes
    array_flips = [np.flip(array, axis=axis) for axis in range(3)]
    mask_flips = [np.flip(mask, axis=axis) for axis in range(3)]

    # apply augmentation 10 times
    for _ in range(10):
        transfo = aug(image=array, mask=mask)
        aug_array = transfo["image"]
        aug_mask = transfo["mask"]

        # check that the augmented array is one of the potential flips
        which_axes = [np.array_equal(aug_array, flip) for flip in array_flips]
        assert any(which_axes)
        img_axis = which_axes.index(True)
        
        # same for the masks
        which_axes = [np.array_equal(aug_mask, flip) for flip in mask_flips]
        assert any(which_axes)
        mask_axis = which_axes.index(True)

        # same flip for array and mask
        assert img_axis == mask_axis
