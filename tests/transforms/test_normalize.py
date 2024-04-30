import numpy as np

from careamics.transforms import Denormalize, Normalize


def test_normalize_denormalize():
    """Test the Normalize transform."""
    # Create data
    array = np.arange(100).reshape((1, 1, 10, 10))

    # Create the transform
    norm = Normalize(
        mean=50,
        std=25,
    )

    # Apply the transform
    normalized: np.array = norm(image=array)["image"]
    assert np.abs(normalized.mean()) < 0.02
    assert np.abs(normalized.std() - 1) < 0.2

    # Create the denormalize transform
    denorm = Denormalize(
        mean=50,
        std=25,
    )

    # Apply the denormalize transform
    denormalized: np.array = denorm(image=normalized)["image"]
    assert np.isclose(denormalized, array).all()
