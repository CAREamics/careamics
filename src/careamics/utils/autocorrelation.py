"""Autocorrelation function."""

import numpy as np
from numpy.typing import NDArray

from .reshape_array import reshape_array


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


def autocorrelation_stack(
    image: NDArray,
    axes: str,
    average_z: bool = True,
) -> NDArray:
    """Compute the autocorrelation of an image stack, averaged over samples.

    The input is normalized to `SC(Z)YX` using `axes`, `autocorrelation` is computed
    per image, and the results are averaged over the sample dimension. Channels are
    kept separate.

    Parameters
    ----------
    image : NDArray
        Input image stack.
    axes : str
        Axes of the input stack, a subset of `STCZYX` containing `Y` and `X`.
    average_z : bool, default=True
        If True, compute a 2D (`YX`) autocorrelation averaged over the sample and
        `Z` dimensions. If False, compute a 3D (`ZYX`) autocorrelation averaged over
        the sample dimension only. Ignored when the data has no `Z` axis.

    Returns
    -------
    numpy.ndarray
        Averaged autocorrelation, `C(Z)YX` with the channel dimension dropped when
        there is a single channel.
    """
    # normalize to canonical SC(Z)YX; also validates axes, Y/X presence, etc.
    reshaped = reshape_array(image, axes)
    has_z = reshaped.ndim == 5
    n_channels = reshaped.shape[1]

    channel_results = []
    for c in range(n_channels):
        # channel is SYX (no Z) or SZYX
        channel = reshaped[:, c]

        if has_z and average_z:
            # treat every (S, Z) slice as an independent 2D realization
            planes = channel.reshape((-1,) + channel.shape[-2:])
        else:
            # 2D per-sample planes (SYX) or 3D per-sample volumes (SZYX)
            planes = channel

        averaged = np.mean([autocorrelation(plane) for plane in planes], axis=0)
        channel_results.append(averaged)

    result = np.stack(channel_results, axis=0)

    if n_channels == 1:
        result = result[0]

    return result
