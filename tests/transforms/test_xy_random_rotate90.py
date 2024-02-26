import pytest

import numpy as np

from careamics.transforms import XYRandomRotate90


def test_randomness(ordered_array):
    """Test randomness of the flipping using the `p` parameter."""
    # create array
    array = ordered_array((1, 2, 2, 1))

    # create augmentation that never applies
    aug = XYRandomRotate90(p=0.)

    # apply augmentation
    augmented = aug(image=array)["image"]
    assert np.array_equal(augmented, array)

    # create augmentation that always applies
    aug = XYRandomRotate90(p=1.)

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
def test_xy_rotate(ordered_array, shape):
    """Test rotation for 2D and 3D arrays."""
    np.random.seed(42)

    # create array
    array: np.ndarray = ordered_array(shape)

    # create augmentation
    is_3D = len(shape) == 4
    aug = XYRandomRotate90(p=1, is_3D=is_3D)

    # potential rotations
    axes = (1, 2) if is_3D else (0, 1)
    rots = [
        np.rot90(array, k=1, axes=axes),
        np.rot90(array, k=2, axes=axes),
        np.rot90(array, k=3, axes=axes),
    ]

    # apply augmentation 10 times
    augs = []
    for _ in range(10):
        augmented = aug(image=array)["image"]

        # check that the augmented array is one of the potential rots
        which_number = [np.array_equal(augmented, rot) for rot in rots]
        
        assert any(which_number)
        augs.append(which_number.index(True))

    # check that all rots were applied (indices of rots)
    assert set(augs) == set((0, 1, 2))


def test_mask_rotate(ordered_array):
    """Test rotating masks in 3D."""
    np.random.seed(42)

    # create array
    array: np.ndarray = ordered_array((2, 2, 2, 4))
    mask = array[..., 2:]
    array = array[..., :2]


    # create augmentation
    is_3D = len(array.shape) == 4
    aug = XYRandomRotate90(p=1, is_3D=is_3D)

    # potential rotations
    axes = (1, 2)
    array_rots = [
        np.rot90(array, k=1, axes=axes),
        np.rot90(array, k=2, axes=axes),
        np.rot90(array, k=3, axes=axes),
    ]
    mask_rots = [
        np.rot90(mask, k=1, axes=axes),
        np.rot90(mask, k=2, axes=axes),
        np.rot90(mask, k=3, axes=axes),
    ]

    # apply augmentation 10 times
    for _ in range(10):
        augmented = aug(image=array, mask=mask)
        aug_array = augmented["image"]
        aug_mask = augmented["mask"]

        # check that the augmented array is one of the potential rots
        which_number = [np.array_equal(aug_array, rot) for rot in array_rots]
        
        assert any(which_number)
        img_n_rots = which_number.index(True)

        # same for the masks
        which_number = [np.array_equal(aug_mask, rot) for rot in mask_rots]
        
        assert any(which_number)
        mask_n_rots = which_number.index(True)

        # same rot for array and mask
        assert img_n_rots == mask_n_rots
        