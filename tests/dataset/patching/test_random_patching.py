import numpy as np
import pytest

from careamics.dataset.patching.random_patching import extract_patches_random


@pytest.mark.parametrize(
    "shape, patch_size",
    [
        ((1, 1, 8, 8), (3, 3)),
        ((1, 3, 8, 8), (3, 3)),
        ((3, 1, 8, 8), (3, 3)),
        ((2, 3, 8, 8), (3, 3)),
        ((1, 1, 5, 8, 8), (3, 3, 3)),
        ((1, 3, 5, 8, 8), (3, 3, 3)),
        ((3, 1, 5, 8, 8), (3, 3, 3)),
        ((2, 3, 5, 8, 8), (3, 3, 3)),
    ],
)
def test_random_patching_unsupervised(ordered_array, shape, patch_size):
    """Check that the patches are extracted correctly.

    Since extract patches is called on already shaped array, dimensions S and C are
    present.
    """
    np.random.seed(42)

    # create array
    array = ordered_array(shape)
    is_3D = len(patch_size) == 3
    top_left = []

    for _ in range(3):
        patch_generator = extract_patches_random(array, patch_size=patch_size)

        # get all patches and targets
        patches = [patch for patch, _ in patch_generator]

        # check patch shape
        for patch in patches:
            # account for C dimension
            assert patch.shape[1:] == patch_size

            # get top_left index in the original array
            if is_3D:
                ind = np.where(array == patch[0, 0, 0, 0])
            else:
                ind = np.where(array == patch[0, 0, 0])

            top_left.append(np.array(ind))

    # check randomness
    coords = np.array(top_left).squeeze()
    assert coords.min() == 0
    assert coords.max() == max(array.shape) - max(patch_size)
    assert len(np.unique(coords, axis=0)) >= 0.7 * np.prod(shape) / np.prod(patch_size)


# @pytest.mark.parametrize(
#     "patch_size",
#     [
#         (2, 2),
#         (4, 2),
#         (4, 8),
#         (8, 8),
#     ],
# )
# def test_extract_patches_random_2d(array_2D, patch_size):
#     """Test extracting patches randomly in 2D."""
#     check_extract_patches_random(array_2D, "SYX", patch_size)


# @pytest.mark.parametrize(
#     "patch_size",
#     [
#         (2, 2),
#         (4, 2),
#         (4, 8),
#         (8, 8),
#     ],
# )
# def test_extract_patches_random_supervised_2d(array_2D, patch_size):
#     """Test extracting patches randomly in 2D."""
#     check_extract_patches_random(
#         array_2D,
#         "SYX",
#         patch_size,
#         target=array_2D
#     )


# @pytest.mark.parametrize(
#     "patch_size",
#     [
#         (2, 2, 4),
#         (4, 2, 2),
#         (2, 8, 4),
#         (4, 8, 8),
#     ],
# )
# def test_extract_patches_random_3d(array_3D, patch_size):
#     """Test extracting patches randomly in 3D.

#     The 3D array is a fixture of shape (1, 8, 16, 16)."""
#     check_extract_patches_random(array_3D, "SZYX", patch_size)
