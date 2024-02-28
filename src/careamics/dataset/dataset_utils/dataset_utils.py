"""Convenience methods for datasets."""
from pathlib import Path
from typing import Callable, List, Tuple, Optional, Union

import numpy as np

from careamics.config.support import SupportedData
from careamics.utils.logging import get_logger

logger = get_logger(__name__)



def get_shape_order(shape_in: Tuple, axes_in: str, ref_axes: str = "STCZYX"):
    """Return the new shape and axes of x, ordered according to the reference axes.

    Parameters
    ----------
    shape_in : Tuple
        Input shape.
    ref_axes : str
        Reference axes.
    axes_in : str
        Input axes.

    Returns
    -------
    Tuple
        New shape.
    str
        New axes.
    Tuple
        Indices of axes in the new axes order.
    """
    indices = [axes_in.find(k) for k in ref_axes]

    # remove all non-existing axes (index == -1)
    indices = tuple(filter(lambda k: k != -1, indices))

    # find axes order and get new shape
    new_axes = [axes_in[ind] for ind in indices]
    new_shape = tuple([shape_in[ind] for ind in indices])

    return new_shape, "".join(new_axes), indices


def list_diff(list1: List, list2: List) -> List:
    """Return the difference between two lists.

    Parameters
    ----------
    list1 : List
        First list.
    list2 : List
        Second list.

    Returns
    -------
    List
        Difference between the two lists.
    """
    return list(set(list1) - set(list2))


def reshape_array(x: np.ndarray, axes: str) -> np.ndarray:
    """Reshape the data to 'SZYXC' or 'SYXC', merging 'S' and 'T' channels if necessary.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    axes : str
        Description of axes in format STCZYX.

    Returns
    -------
    np.ndarray
        Reshaped array.
    """
    _x = x
    _axes = axes

    # sanity checks
    if len(_axes) != len(_x.shape):
        raise ValueError(
            f"Incompatible data shape ({_x.shape}) and axes ({_axes})."
        )

    # get new x shape
    new_x_shape, new_axes, indices = get_shape_order(_x.shape, _axes)

    # if S is not in the list of axes, then add a singleton S
    if "S" not in new_axes:
        new_axes = "S" + new_axes
        _x = _x[np.newaxis, ...]
        new_x_shape = (1,) + new_x_shape

        # need to change the array of indices
        indices = [0] + [1 + i for i in indices]

    # reshape by moving axes
    destination = list(range(len(indices)))
    _x = np.moveaxis(_x, indices, destination)

    # remove T if necessary
    if "T" in new_axes:
        new_x_shape = (-1,) + new_x_shape[2:]  # remove T and S
        new_axes = new_axes.replace("T", "")

        # reshape S and T together
        _x = _x.reshape(new_x_shape)

    # add channel
    if "C" not in new_axes:
        # Add channel axis after S
        _x = np.expand_dims(_x, new_axes.index("S") + 1)

    return _x
