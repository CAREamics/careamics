import numpy as np

from careamics.dataset_ng.patch_filter import MaxPatchFilter


def test_csbdeep_filter():
    """Test MaxPatchFilter functionality."""
    size = 16
    img = np.arange(size * size).reshape((size, size))
    img[size // 4 : -size // 4, size // 4 : -size // 4] = 255
    patch = img[4:12, 4:12]

    corner_val = patch[0, 0]

    # raise threshold over corner value, should be filtered out
    max_filter = MaxPatchFilter(threshold=corner_val + 10)
    assert max_filter.filter_out(patch)

    # threshold below corner value, should be kept (return val = False)
    max_filter = MaxPatchFilter(threshold=corner_val - 10)
    assert not max_filter.filter_out(patch)
