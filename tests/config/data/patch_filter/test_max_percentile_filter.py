import numpy as np

from careamics.dataset_ng.patch_filter import MaxPercentilePatchFilter


def test_filter():
    """Test MaskPatchFilter functionality."""
    size = 16
    img = np.zeros((size, size))
    img[size // 4 : -size // 4, size // 4 : -size // 4] = 255

    mask_filter = MaxPercentilePatchFilter(
        max_value=255,
        weight=0.75,
    )

    assert mask_filter.filter_out(img[0:4, 0:4])  # no overlap with signal
    assert mask_filter.filter_out(img[2:6, 2:6])  # 25% overlap
    assert not mask_filter.filter_out(img[2:6, 4:8])  # 50% overlap
    assert not mask_filter.filter_out(img[4:8, 4:8])  # 100% overlap
