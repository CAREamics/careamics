import numpy as np
import pytest

from careamics_restoration.metrics import (
    MetricTracker,
    scale_invariant_psnr,
    zero_mean,
)


@pytest.mark.parametrize(
    "x",
    [
        5.6,
        np.array([1, 2, 3, 4, 5]),
        np.array([[1, 2, 3], [4, 5, 6]]),
    ],
)
def test_zero_mean(x):
    assert np.allclose(zero_mean(x), x - np.mean(x))


@pytest.mark.parametrize(
    "gt, pred, result",
    [
        (np.array([1, 2, 3, 4, 5, 6]), np.array([1, 2, 3, 4, 5, 6]), 332.22),
        (np.array([[1, 2, 3], [4, 5, 6]]), np.array([[1, 2, 3], [4, 5, 6]]), 332.22),
    ],
)
def test_scale_invariant_psnr(gt, pred, result):
    assert scale_invariant_psnr(gt, pred) == pytest.approx(result, rel=5e-3)


def test_metric_tracker():
    tracker = MetricTracker()

    # check initial state
    assert tracker.sum == 0
    assert tracker.count == 0
    assert tracker.avg == 0
    assert tracker.val == 0

    # run a few updates
    n = 5
    for i in range(n):
        tracker.update(i, n)

    # check values
    assert tracker.sum == n * (n * (n - 1)) / 2
    assert tracker.count == n * n
    assert tracker.avg == (n - 1) / 2
    assert tracker.val == n - 1
