import numpy as np

from careamics.utils import autocorrelation


def test_autocorrelation():
    """Test that the autocorrelation is normalized to the zero-shift value and that this
    value is centered."""
    rng = np.random.default_rng(42)

    # create a random image
    image = rng.random((5, 5))
    center = 2

    # compute autocorrelation
    autocorr = autocorrelation(image)
    assert autocorr.shape == image.shape

    # check that the max value is 1
    assert np.isclose(autocorr.max(), 1.0, atol=1e-6)

    # check that the zero-shift value is 1
    assert np.isclose(autocorr[center, center], 1.0, atol=1e-6)


def test_correlation():
    """Test the autocorrelation of an image with bands."""
    image = np.zeros((10, 10))
    for i in range(image.shape[0]):
        if i % 2 == 0:
            image[i] = 1

    # compute autocorrelation
    autocorr = autocorrelation(image)
    assert autocorr.shape == image.shape

    # check that each band maximizes the autocorrelation
    for i in range(autocorr.shape[0]):
        if i % 2 == 0:
            assert np.allclose(autocorr[i], -1.0)
        else:
            assert np.allclose(autocorr[i], 1.0)
