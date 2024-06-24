import numpy as np
import pytest

from careamics.dataset.dataset_utils.running_stats import compute_normalization_stats


@pytest.mark.parametrize("samples, channels", [[1, 2], [1, 2]])
def test_compute_normalization_stats(samples, channels):
    """Test the compute_normalization_stats function."""
    # Create data
    array = np.arange(100 * samples * channels).reshape((samples, channels, 10, 10))

    # Compute mean and std
    mean, std = compute_normalization_stats(image=array)
    for ch in range(array.shape[1]):
        assert np.isclose(mean[ch], array[:, ch, ...].mean())
        assert np.isclose(std[ch], array[:, ch, ...].std())

    # Create data 3D
    array = np.arange(1000 * samples * channels).reshape(
        (samples, channels, 10, 10, 10)
    )

    # Compute mean and std
    mean, std = compute_normalization_stats(image=array)
    for ch in range(array.shape[1]):
        assert np.isclose(mean[ch], array[:, ch, ...].mean())
        assert np.isclose(std[ch], array[:, ch, ...].std())
