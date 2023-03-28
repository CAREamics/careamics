import numpy as np


def augment_single(image, seed=1, channel_dim=True):
    """Augment single data object(2D or 3D) before batching by rotating and flipping patches.

    Parameters
    ----------
    patches : np.ndarray
        array containing single image or patch, 2D or 3D
    seed : int, optional
        seed for random number generator, controls the rotation and flipping
    channel_dim : bool, optional
        Set to True if the channel dimension is present, by default True

    Returns
    -------
    np.ndarray
        _description_
    """
    rotate_state = np.random.randint(0, 5)
    flip_state = np.random.randint(0, 2)
    rotated = np.rot90(image, k=rotate_state, axes=(-2, -1))
    flipped = np.flip(rotated, axis=-1) if flip_state == 1 else rotated
    # TODO check for memory leak
    return flipped.copy()