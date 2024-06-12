import numpy as np
import pytest

from careamics.dataset.dataset_utils import compute_normalization_stats
from careamics.transforms import Denormalize, Normalize


@pytest.mark.parametrize("channels", [1, 2])
def test_normalize_denormalize(channels):
    """Test the Normalize transform."""
    # Create data, adding sample dimension for stats computation
    array = np.arange(100 * channels).reshape((1, channels, 10, 10))

    # Compute mean and std
    means, stds = compute_normalization_stats(image=array)

    # Create the transform
    norm = Normalize(
        image_means=means,
        image_stds=stds,
    )

    # Apply the transform, removing the sample dimension
    normalized, _ = norm(patch=array[0])
    assert np.abs(normalized.mean()) < 0.02
    assert np.abs(normalized.std() - 1) < 0.2

    # Create the denormalize transform
    denorm = Denormalize(
        image_means=means,
        image_stds=stds,
    )

    # Apply the denormalize transform
    denormalized = denorm(patch=normalized[np.newaxis, ...])  # need to add batch dim
    assert np.isclose(denormalized, array, atol=1e-6).all()
