import numpy as np
from typing import Tuple


# TODO channel_dim is unused
# TODO document...
def augment_batch(
    patch: np.ndarray,
    orig_image: np.ndarray,  # TODO what is that for?
    mask: np.ndarray,
    seed=1,
    channel_dim: bool = True,
) -> Tuple[np.ndarray, ...]:
    """Augment a single array by applying a random 90 degress rotation and
    possibly a flip.

    Parameters
    ----------
    patch : np.ndarray
        Array containing single image or patch, 2D or 3D
    seed : int, optional
        Seed for random number generator, controls the rotation and flipping
    channel_dim : bool, optional
        Set to True if the channel dimension is present, by default True

    Returns
    -------
    np.ndarray
        _description_
    """
    # np.random.seed(seed)
    rng = np.random.default_rng(seed=seed)
    rotate_state = rng.integers(0, 4)
    flip_state = rng.integers(0, 2)
    return (
        flip_and_rotate(patch, rotate_state, flip_state),
        flip_and_rotate(orig_image, rotate_state, flip_state),
        flip_and_rotate(mask, rotate_state, flip_state),
    )


def flip_and_rotate(
    image: np.ndarray, rotate_state: int, flip_state: int
) -> np.ndarray:
    """Apply the given number of 90 degrees rotations and flip to an array.

    Parameters
    ----------
    image : np.ndarray
        array containing single image or patch, 2D or 3D
    rotate_state : int
        number of 90 degree rotations to apply
    flip_state : int
        0 or 1, whether to flip the array or not

    Returns
    -------
    np.ndarray
        Flipped and rotated array
    """
    rotated = np.rot90(image, k=rotate_state, axes=(-2, -1))
    flipped = np.flip(rotated, axis=-1) if flip_state == 1 else rotated
    return flipped.copy()
