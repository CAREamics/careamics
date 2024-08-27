import numpy as np
import pytest

from careamics.utils.metrics import (
    _zero_mean,
    scale_invariant_psnr,
    psnr
)

# TODO: add missing test cases
# - `psnr` result
# - `_fix` and `fix_range`
# - `RunningPSNR`


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


@pytest.mark.parametrize(
    "data_type",
    [
        np.uint8,
        np.uint16,
        np.float32,
        np.int64
    ],
)
def test_psnr(data_type: np.dtype):
    gt_ = np.random.randint(0, 255, size=(16, 16)).astype(data_type)
    pred_ = np.random.randint(0, 255, size=(16, 16)).astype(data_type)
    if data_type != np.float32:
        assert psnr(gt_, pred_, None) is not None
    assert psnr(gt_, pred_, 255) is not None
    range_ = gt_.max() - gt_.min()
    assert psnr(gt_, pred_, range_) is not None
    # add check on the result