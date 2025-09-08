import numpy as np

from careamics.dataset_ng.patch_filter import MaxPercentilePatchFilter


def test_csbdeep_filter():
    """Test MaxPercentilePatchFilter functionality."""
    size = 16
    img = np.arange(size * size).reshape((size, size))
    img[size // 4 : -size // 4, size // 4 : -size // 4] = 255
    patch = img[4:12, 4:12]

    corner_val = patch[0, 0]

    # raise threshold over corner value, should be filtered out
    max_filter = MaxPercentilePatchFilter(
        max_value=255,
        weight=0.05 + corner_val / 255,
    )
    assert max_filter.filter_out(patch)

    # threshold below corner value, should be kept (return val = False)
    max_filter = MaxPercentilePatchFilter(
        max_value=255,
        weight=-0.05 + corner_val / 255,
    )
    assert not max_filter.filter_out(patch)
