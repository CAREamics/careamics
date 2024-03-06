import numpy as np
import pytest

from careamics.dataset.patching.validate_patch_dimension import (
    validate_patch_dimensions,
)


@pytest.mark.parametrize(
    "arr_shape, patch_size",
    [
        ((1, 1, 8, 8), (2, 2)),
        ((1, 1, 8, 8, 8), (2, 2, 2)),
    ],
)
def test_patches_sanity_check(arr_shape, patch_size):
    arr = np.zeros(arr_shape)
    is_3d_patch = len(patch_size) == 3
    # check if the patch is 2D or 3D. Subtract 1 because the first dimension is sample
    validate_patch_dimensions(arr, patch_size, is_3d_patch)


@pytest.mark.parametrize(
    "arr_shape, patch_size",
    [
        # Wrong number of dimensions 2D
        # minimum 3 dimensions CYX
        ((10, 10), (5, 5, 5)),
        # Wrong number of dimensions 3D
        ((1, 1, 10, 10, 10), (5, 5)),
        # Wrong z patch size
        ((1, 10, 10), (5, 5, 5)),
        ((10, 10, 10), (10, 5, 5)),
        # Wrong YX patch sizes
        ((1, 10, 10), (12, 5)),
        ((1, 10, 10), (5, 11)),
    ],
)
def test_patches_sanity_check_invalid_cases(arr_shape, patch_size):
    arr = np.zeros(arr_shape)
    is_3d_patch = len(patch_size) == 3
    # check if the patch is 2D or 3D. Subtract 1 because the first dimension is sample
    with pytest.raises(ValueError):
        validate_patch_dimensions(arr, patch_size, is_3d_patch)
