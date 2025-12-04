import pytest
import numpy as np

from careamics.dataset_ng.normalization.running_quantile import QuantileEstimator


@pytest.mark.parametrize(
    "channels, lower_quantiles, upper_quantiles",
    [
        (1, [0.01], [0.99]),
        (2, [0.1, 0.2], [0.8, 0.9]),
        (3, [0.05, 0.1, 0.25], [0.75, 0.9, 0.95]),
    ],
)
def test_exact_mode(channels, lower_quantiles, upper_quantiles):
    """Test exact quantile computation for small datasets."""
    np.random.seed(42)
    estimator = QuantileEstimator(
        lower_quantiles=lower_quantiles, upper_quantiles=upper_quantiles
    )
    data = np.random.rand(channels, 64, 64).astype(np.float32) * 1000

    estimator.update(data)
    lower, upper = estimator.finalize()

    for ch in range(channels):
        assert lower[ch] == np.quantile(data[ch], lower_quantiles[ch])
        assert upper[ch] == np.quantile(data[ch], upper_quantiles[ch])


@pytest.mark.parametrize(
    "n_patches, exact_threshold",
    [
        (5, 1000),
        (10, 500),
        (20, 100),
    ],
)
def test_histogram_mode_accuracy(n_patches, exact_threshold):
    """Test histogram-based estimation for large datasets."""
    np.random.seed(42)
    estimator = QuantileEstimator(
        lower_quantiles=[0.01],
        upper_quantiles=[0.99],
        exact_threshold=exact_threshold,
    )
    all_data = []
    for _ in range(n_patches):
        patch = np.random.rand(1, 64, 64).astype(np.float32) * 65000
        estimator.update(patch)
        all_data.append(patch)

    lower, upper = estimator.finalize()
    all_data = np.concatenate(all_data, axis=1)

    expected_lower = np.quantile(all_data, 0.01)
    expected_upper = np.quantile(all_data, 0.99)
    assert np.isclose(lower[0], expected_lower, atol=10)
    assert np.isclose(upper[0], expected_upper, atol=10)


@pytest.mark.parametrize("constant_value", [0.0, 42.0, -100.0, 65535.0])
def test_constant_data(constant_value):
    """Test handling of constant data (zero range)."""
    estimator = QuantileEstimator(
        lower_quantiles=[0.1], upper_quantiles=[0.9]
    )
    data = np.full((1, 32, 32), constant_value, dtype=np.float32)

    estimator.update(data)
    lower, upper = estimator.finalize()

    assert lower[0] == constant_value
    assert upper[0] == constant_value


def test_no_data_returns_zero():
    """Test that finalize returns zeros when no data added."""
    estimator = QuantileEstimator(
        lower_quantiles=[0.1], upper_quantiles=[0.9]
    )
    lower, upper = estimator.finalize()

    assert lower[0] == 0.0
    assert upper[0] == 0.0


@pytest.mark.parametrize("n_updates", [2, 5, 10])
def test_streaming_multiple_updates(n_updates):
    """Test accumulating data across multiple updates."""
    np.random.seed(42)
    estimator = QuantileEstimator(
        lower_quantiles=[0.1], upper_quantiles=[0.9]
    )

    patches = [np.random.rand(1, 32, 32).astype(np.float32) * 100 for _ in range(n_updates)]
    for patch in patches:
        estimator.update(patch)

    lower, upper = estimator.finalize()

    all_data = np.concatenate(patches, axis=1)
    assert lower[0] == np.quantile(all_data, 0.1)
    assert upper[0] == np.quantile(all_data, 0.9)
