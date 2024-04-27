import numpy as np
import pytest

from careamics.utils.metrics import (
    _zero_mean,
    scale_invariant_psnr,
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
    assert np.allclose(_zero_mean(x), x - np.mean(x))


@pytest.mark.parametrize(
    "gt, pred, result",
    [
        (np.array([1, 2, 3, 4, 5, 6]), np.array([1, 2, 3, 4, 5, 6]), 332.22),
        (np.array([[1, 2, 3], [4, 5, 6]]), np.array([[1, 2, 3], [4, 5, 6]]), 332.22),
    ],
)
def test_scale_invariant_psnr(gt, pred, result):
    assert scale_invariant_psnr(gt, pred) == pytest.approx(result, rel=5e-3)
