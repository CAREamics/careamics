import numpy as np


# TODO channel_dim is unused
def augment_single(image: np.ndarray, seed=1, channel_dim: bool = True) -> np.ndarray:
    """Augment a single array by applying a random 90 degress rotation and
    possibly a flip.

    Parameters
    ----------
    image : np.ndarray
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
    # np.random.seed(seed)

    rotate_state = np.random.randint(0, 5)
    flip_state = np.random.randint(0, 2)
    rotated = np.rot90(image, k=rotate_state, axes=(-2, -1))
    flipped = np.flip(rotated, axis=-1) if flip_state == 1 else rotated

    # TODO check for memory leak

    return flipped.copy()
