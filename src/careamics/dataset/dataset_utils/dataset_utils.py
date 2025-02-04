"""Dataset utilities."""

import numpy as np

from careamics.utils.logging import get_logger

logger = get_logger(__name__)


def _get_shape_order(
    shape_in: tuple[int, ...], axes_in: str, ref_axes: str = "STCZYX"
) -> tuple[tuple[int, ...], str, list[int]]:
    """
    Compute a new shape for the array based on the reference axes.

    Parameters
    ----------
    shape_in : tuple[int, ...]
        Input shape.
    axes_in : str
        Input axes.
    ref_axes : str
        Reference axes.

    Returns
    -------
    tuple[tuple[int, ...], str, list[int]]
        New shape, new axes, indices of axes in the new axes order.
    """
    indices = [axes_in.find(k) for k in ref_axes]

    # remove all non-existing axes (index == -1)
    new_indices = list(filter(lambda k: k != -1, indices))

    # find axes order and get new shape
    new_axes = [axes_in[ind] for ind in new_indices]
    new_shape = tuple([shape_in[ind] for ind in new_indices])

    return new_shape, "".join(new_axes), new_indices


def reshape_array(x: np.ndarray, axes: str) -> np.ndarray:
    """Reshape the data to (S, C, (Z), Y, X) by moving axes.

    If the data has both S and T axes, the two axes will be merged. A singleton
    dimension is added if there are no C axis.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    axes : str
        Description of axes in format `STCZYX`.

    Returns
    -------
    np.ndarray
        Reshaped array with shape (S, C, (Z), Y, X).
    """
    _x = x
    _axes = axes

    # sanity checks
    if len(_axes) != len(_x.shape):
        raise ValueError(
            f"Incompatible data shape ({_x.shape}) and axes ({_axes}). Are the axes "
            f"correct?"
        )

    # get new x shape
    new_x_shape, new_axes, indices = _get_shape_order(_x.shape, _axes)

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
