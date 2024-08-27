import numpy as np
import pytest

from careamics.utils.metrics import (
    _zero_mean,
    scale_invariant_psnr,
)


@pytest.mark.parametrize(
    "x",
    [
        np.array([1, 2, 3, 4, 5]),
        np.array([[1, 2, 3], [4, 5, 6]]),
    ],
)
def test_zero_mean(x: np.ndarray):
    assert np.allclose(_zero_mean(x), x - np.mean(x))


# TODO: with 2 identical arrays, shouldn't the result be `inf`?
@pytest.mark.parametrize(
    "gt, pred, result",
    [
        (np.array([1, 2, 3, 4, 5, 6]), np.array([1, 2, 3, 4, 5, 6]), 332.22),
        (np.array([[1, 2, 3], [4, 5, 6]]), np.array([[1, 2, 3], [4, 5, 6]]), 332.22),
    ],
)
def test_scale_invariant_psnr(gt: np.ndarray, pred: np.ndarray, result: float):
    assert scale_invariant_psnr(gt, pred) == pytest.approx(result, rel=5e-3)


