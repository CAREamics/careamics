"""Autocorrelation function."""

import numpy as np
from numpy.typing import NDArray


def autocorrelation(image: NDArray) -> NDArray:
    """Compute the autocorrelation of an image.

    This method is used to explore spatial correlations in images,
    in particular in the noise.

    The autocorrelation is normalized to the zero-shift value, which is centered in
    the resulting images.

    Parameters
    ----------
    image : NDArray
        Input image.

    Returns
    -------
    numpy.ndarray
        Autocorrelation of the input image.
    """
    # normalize image
    image = (image - np.mean(image)) / np.std(image)

    # compute autocorrelation in fourier space
    image = np.fft.fftn(image)
    image = np.abs(image) ** 2
    image = np.fft.ifftn(image).real

    # normalize to zero shift value
    image = image / image.flat[0]

    # shift zero frequency to center
    image = np.fft.fftshift(image)

    return image
