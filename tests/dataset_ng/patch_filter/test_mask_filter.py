import numpy as np

from careamics.dataset_ng.patch_filter import MaskFilter

# TODO test multiple patches at once


def test_filter():
    """Test MaskFilter functionality."""
    size = 16
    mask = np.zeros((size, size))
    mask[size // 4 : -size // 4, size // 4 : -size // 4] = 1

    mask_filter = MaskFilter(coverage=0.50)

    # Create mask patches to test
    assert mask_filter.filter_out(mask[0:4, 0:4])  # corner, no overlap with mask
    assert mask_filter.filter_out(mask[2:6, 2:6])  # 25% overlap (4 pixels out of 16)
    assert not mask_filter.filter_out(
        mask[2:6, 4:8]
    )  # 50% overlap (8 pixels out of 16)
    assert not mask_filter.filter_out(
        mask[4:8, 4:8]
    )  # 100% overlap (16 pixels out of 16)
