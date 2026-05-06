"""Shared utilities for normalization."""

import numpy as np
from numpy.typing import NDArray


def broadcast_stats(stats: list[float], n_channels: int, name: str) -> list[float]:
    """Broadcast a length-1 stats list to *n_channels*, or validate length.

    Parameters
    ----------
    stats : list[float]
        A length-1 list (global) or a per-channel list.
    n_channels : int
        Expected number of channels.
    name : str
        Field name used in error messages.

    Returns
    -------
    list[float]
        Values matching *n_channels*.

    Raises
    ------
    ValueError
        If *stats* length is neither 1 nor *n_channels*.
    """
    if len(stats) == 1:
        return stats * n_channels
    if len(stats) != n_channels:
        raise ValueError(
            f"Number of {name} ({len(stats)}) and number of channels "
            f"({n_channels}) do not match."
        )
    return stats


def reshape_stats(stats: list[float], ndim: int, channel_axis: int = 0) -> NDArray:
    """Reshape stats to match the number of dimensions of the input image.

    Parameters
    ----------
    stats : list of float
        List of per-channel statistic values.
    ndim : int
        Number of dimensions in the image.
    channel_axis : int, optional
        Position of the channel axis in the image.

    Returns
    -------
    NDArray
        Stats array reshaped, with the channel dimension
        at the specified axis and singleton dimensions elsewhere.
    """
    shape = [1] * ndim
    shape[channel_axis] = len(stats)
    return np.array(stats, dtype=np.float32).reshape(shape)
