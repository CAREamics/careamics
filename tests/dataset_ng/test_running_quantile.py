import numpy as np
import pytest

from careamics.dataset_ng.normalization.running_quantile import QuantileEstimator

QUANTILE_PAIRS = [(0.01, 0.99), (0.05, 0.95), (0.1, 0.9)]


@pytest.mark.parametrize("n_patches", [5, 10, 20])
@pytest.mark.parametrize("lower_q, upper_q", QUANTILE_PAIRS)
def test_histogram_accuracy(n_patches, lower_q, upper_q):
    np.random.seed(42)
    estimator = QuantileEstimator(
        lower_quantiles=[lower_q],
        upper_quantiles=[upper_q],
    )
    all_data = []
    for _ in range(n_patches):
        patch = np.random.rand(1, 64, 64).astype(np.float32) * 65000
        estimator.update(patch)
        all_data.append(patch)

    lower, upper = estimator.finalize()
    all_data = np.concatenate(all_data, axis=1)

    assert np.isclose(lower[0], np.quantile(all_data, lower_q), rtol=0.01)
    assert np.isclose(upper[0], np.quantile(all_data, upper_q), rtol=0.01)


@pytest.mark.parametrize("lower_q, upper_q", QUANTILE_PAIRS)
def test_microscopy_like_distribution(lower_q, upper_q):
    np.random.seed(42)
    estimator = QuantileEstimator(lower_quantiles=[lower_q], upper_quantiles=[upper_q])

    all_patches = []
    for _ in range(50):
        patch = np.random.exponential(scale=50, size=(1, 64, 64)).astype(np.float32)
        n_spots = np.random.randint(1, 5)
        for _ in range(n_spots):
            y, x = np.random.randint(0, 64, 2)
            patch[0, y, x] = np.random.uniform(500, 1000)
        estimator.update(patch)
        all_patches.append(patch)

    lower, upper = estimator.finalize()
    all_data = np.concatenate(all_patches, axis=1)

    assert np.isclose(lower[0], np.quantile(all_data, lower_q), rtol=0.05)
    assert np.isclose(upper[0], np.quantile(all_data, upper_q), rtol=0.05)


@pytest.mark.parametrize("lower_q, upper_q", QUANTILE_PAIRS)
def test_bimodal_distribution(lower_q, upper_q):
    np.random.seed(42)
    estimator = QuantileEstimator(lower_quantiles=[lower_q], upper_quantiles=[upper_q])

    all_patches = []
    for _ in range(30):
        background = np.random.normal(loc=100, scale=10, size=(1, 32, 32))
        foreground = np.random.normal(loc=500, scale=50, size=(1, 32, 32))
        mask = np.random.random((1, 32, 32)) > 0.7
        patch = np.where(mask, foreground, background).astype(np.float32)
        estimator.update(patch)
        all_patches.append(patch)

    lower, upper = estimator.finalize()
    all_data = np.concatenate(all_patches, axis=1)

    assert np.isclose(lower[0], np.quantile(all_data, lower_q), rtol=0.05)
    assert np.isclose(upper[0], np.quantile(all_data, upper_q), rtol=0.05)


@pytest.mark.parametrize("lower_q, upper_q", QUANTILE_PAIRS)
def test_16bit_dynamic_range(lower_q, upper_q):
    np.random.seed(42)
    estimator = QuantileEstimator(lower_quantiles=[lower_q], upper_quantiles=[upper_q])

    all_patches = []
    for _ in range(20):
        patch = np.random.randint(0, 65535, size=(1, 64, 64)).astype(np.float32)
        estimator.update(patch)
        all_patches.append(patch)

    lower, upper = estimator.finalize()
    all_data = np.concatenate(all_patches, axis=1)

    assert np.isclose(lower[0], np.quantile(all_data, lower_q), atol=100)
    assert np.isclose(upper[0], np.quantile(all_data, upper_q), atol=100)


def test_multichannel_different_quantiles():
    np.random.seed(42)
    estimator = QuantileEstimator(
        lower_quantiles=[0.01, 0.05, 0.1],
        upper_quantiles=[0.99, 0.95, 0.9],
    )

    all_patches = []
    for _ in range(25):
        patch = np.random.rand(3, 32, 32).astype(np.float32) * 1000
        estimator.update(patch)
        all_patches.append(patch)

    lower, upper = estimator.finalize()
    all_data = np.concatenate(all_patches, axis=1)

    for ch, (lq, uq) in enumerate(
        zip([0.01, 0.05, 0.1], [0.99, 0.95, 0.9], strict=True)
    ):
        assert np.isclose(lower[ch], np.quantile(all_data[ch], lq), rtol=0.05)
        assert np.isclose(upper[ch], np.quantile(all_data[ch], uq), rtol=0.05)


@pytest.mark.parametrize("lower_q, upper_q", QUANTILE_PAIRS)
def test_rebinning_with_expanding_range(lower_q, upper_q):
    np.random.seed(42)
    estimator = QuantileEstimator(
        lower_quantiles=[lower_q],
        upper_quantiles=[upper_q],
    )

    all_patches = []
    for _ in range(10):
        patch = np.random.rand(1, 32, 32).astype(np.float32) * 100
        estimator.update(patch)
        all_patches.append(patch)

    for _ in range(10):
        patch = np.random.rand(1, 32, 32).astype(np.float32) * 1000
        estimator.update(patch)
        all_patches.append(patch)

    lower, upper = estimator.finalize()
    all_data = np.concatenate(all_patches, axis=1)

    assert np.isclose(lower[0], np.quantile(all_data, lower_q), rtol=0.1)
    assert np.isclose(upper[0], np.quantile(all_data, upper_q), rtol=0.1)


@pytest.mark.parametrize("constant_value", [0.0, 42.0, -100.0, 65535.0])
@pytest.mark.parametrize("lower_q, upper_q", QUANTILE_PAIRS)
def test_constant_data(constant_value, lower_q, upper_q):
    estimator = QuantileEstimator(lower_quantiles=[lower_q], upper_quantiles=[upper_q])
    data = np.full((1, 32, 32), constant_value, dtype=np.float32)

    estimator.update(data)
    lower, upper = estimator.finalize()

    assert np.isclose(lower[0], constant_value, atol=1e-5)
    assert np.isclose(upper[0], constant_value, atol=1e-5)


def test_no_data_returns_zero():
    estimator = QuantileEstimator(lower_quantiles=[0.1], upper_quantiles=[0.9])
    lower, upper = estimator.finalize()

    assert lower[0] == 0.0
    assert upper[0] == 0.0
