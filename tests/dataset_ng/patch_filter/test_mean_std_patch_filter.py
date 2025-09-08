import numpy as np

from careamics.dataset_ng.patch_filter import MeanStdPatchFilter


def test_meanstd_filter():
    """Test ShannonEntropyFilter functionality."""
    size = 8
    img = np.zeros((size, size))
    img[size // 2 :, size // 2 :] = 255

    # test mean filtering
    meanstd_filter = MeanStdPatchFilter(mean_threshold=255 / 2)

    assert meanstd_filter.filter_out(img[0:4, 0:4])  # corner, 0 mean
    assert meanstd_filter.filter_out(img[2:6, 2:6])  # quarter bg
    assert not meanstd_filter.filter_out(img[2:6, 4:])  # half bg
    assert not meanstd_filter.filter_out(img[4:, 4:])  # full fg

    # filter on std
    meanstd_filter.std_threshold = 50

    assert meanstd_filter.filter_out(img[0:4, 0:4])  # corner, null std
    assert meanstd_filter.filter_out(
        img[2:6, 2:6]
    )  # quarter bg, mean does not pass threshold
    assert not meanstd_filter.filter_out(img[2:6, 4:])  # half bg, with high std
    assert meanstd_filter.filter_out(img[4:, 4:])  # full fg, null std
