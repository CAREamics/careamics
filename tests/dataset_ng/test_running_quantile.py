import pytest

import numpy as np

from careamics.dataset_ng.normalization.running_quantile import QuantileEstimator


@pytest.mark.parametrize(
    "samples, channels, lower_quantiles, upper_quantiles",
    [
        (1, 1, [0.01], [0.99]),
        (1, 3, [0.01, 0.1, 0.2], [0.8, 0.9, 0.99]),
        (3, 1, [0.01], [0.99]),
        (3, 3, [0.01, 0.1, 0.2], [0.8, 0.9, 0.99]),
    ]
)
def test_compute_normalization_stats(
    samples, channels, lower_quantiles, upper_quantiles
):
    test_arr = np.random.randint(0, 256, (samples, channels, 10, 10))
    test_arr = test_arr.astype(np.float32)

    estimator = QuantileEstimator(
        lower_quantiles=lower_quantiles, upper_quantiles=upper_quantiles
    )

    for arr in test_arr:
        estimator.update(arr)

    computed_lower_quantiles, computed_upper_quantiles = estimator.finalize()

    assert computed_lower_quantiles.shape == (channels,)
    assert computed_upper_quantiles.shape == (channels,)

    expected_lower_quantiles = []
    expected_upper_quantiles = []
    for ch in range(channels):
        channel_data = test_arr[:, ch, ...]
        expected_lower_quantile = np.quantile(channel_data, q=lower_quantiles[ch])
        expected_upper_quantile = np.quantile(channel_data, q=upper_quantiles[ch])
        expected_lower_quantiles.append(expected_lower_quantile)
        expected_upper_quantiles.append(expected_upper_quantile)
    expected_lower_quantiles = np.array(expected_lower_quantiles)
    expected_upper_quantiles = np.array(expected_upper_quantiles)

    assert np.allclose(computed_lower_quantiles, expected_lower_quantiles, atol=1e-1)
    assert np.allclose(computed_upper_quantiles, expected_upper_quantiles, atol=1e-1)
