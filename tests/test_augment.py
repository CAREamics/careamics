import numpy as np
import pytest
from careamics_restoration.augment import augment_single

ARRAY_2D = np.array([[1, 2, 3], [4, 5, 6]])
AUG_ARRAY_2D = [
    np.array([[3, 6], [2, 5], [1, 4]]),  # rot90
    np.array([[6, 3], [5, 2], [4, 1]]),  # rot90 + flip
    np.array([[6, 5, 4], [3, 2, 1]]),  # rot180
    np.array([[4, 5, 6], [1, 2, 3]]),  # rot180 + flip
    np.array([[4, 1], [5, 2], [6, 3]]),  # rot270
    np.array([[1, 4], [2, 5], [3, 6]]),  # rot270 + flip
]
ARRAY_3D = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
AUG_ARRAY_3D = [
    np.array([[[3, 6], [2, 5], [1, 4]], [[9, 12], [8, 11], [7, 10]]]),  # rot90
    np.array([[[6, 3], [5, 2], [4, 1]], [[12, 9], [11, 8], [10, 7]]]),  # rot90 + flip
    np.array([[[6, 5, 4], [3, 2, 1]], [[12, 11, 10], [9, 8, 7]]]),  # rot180
    np.array([[[4, 5, 6], [1, 2, 3]], [[10, 11, 12], [7, 8, 9]]]),  # rot180 + flip
    np.array([[[4, 1], [5, 2], [6, 3]], [[10, 7], [11, 8], [12, 9]]]),  # rot270
    np.array([[[1, 4], [2, 5], [3, 6]], [[7, 10], [8, 11], [9, 12]]]),  # rot270 + flip
]


@pytest.mark.parametrize(
    "array, possible_augmentations",
    [
        (ARRAY_2D, AUG_ARRAY_2D),
        (ARRAY_3D, AUG_ARRAY_3D),
    ],
)
def test_augment_single(array, possible_augmentations):
    for i in range(5):
        aug_array = augment_single(array, seed=i)

        array_found = False
        for aug_possible in possible_augmentations:
            array_found = array_found or np.array_equal(aug_array, aug_possible)
