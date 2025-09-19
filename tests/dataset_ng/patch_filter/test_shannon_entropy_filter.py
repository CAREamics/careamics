import numpy as np
from skimage.measure import shannon_entropy

from careamics.dataset_ng.patch_filter import ShannonPatchFilter


def test_shannon_filter():
    """Test ShannonEntropyFilter functionality."""
    rng = np.random.default_rng(42)

    size = 8
    img = np.zeros((size, size))
    img[size // 2 :, size // 2 :] = rng.integers(100, 255, (size // 2, size // 2))

    corner_entropy = shannon_entropy(img[size // 2 :, size // 2 :])

    shannon_filter = ShannonPatchFilter(threshold=corner_entropy / 2)

    # exclude border
    assert shannon_filter.filter_out(img[0:4, 0:4])  # corner, no entropy

    # exclude low entropy
    assert shannon_filter.filter_out(img[2:6, 2:6])  # quarter bg, low entropy

    # include higher entropy
    assert not shannon_filter.filter_out(img[2:6, 4:])  # half bg, higher entropy

    # include high entropy
    assert not shannon_filter.filter_out(img[4:, 4:])  # full fg, high entropy
