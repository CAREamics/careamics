import numpy as np
from typing import Tuple


# TODO channel_dim is unused
def augment_batch(
    patch: np.ndarray,
    orig_image: np.ndarray,
    mask: np.ndarray,
    seed=1,
    channel_dim: bool = True,
) -> Tuple[np.ndarray]:
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
    rng = np.random.default_rng(seed=seed)
    rotate_state = rng.integers(0, 5)
    flip_state = rng.integers(0, 2)
    return (
        augment_single(patch, rotate_state, flip_state),
        augment_single(orig_image, rotate_state, flip_state),
        augment_single(mask, rotate_state, flip_state),
    )


def augment_single(image: np.ndarray, rotate_state: int, flip_state: int) -> np.ndarray:
    rotated = np.rot90(image, k=rotate_state, axes=(-2, -1))
    flipped = np.flip(rotated, axis=-1) if flip_state == 1 else rotated
    return flipped.copy()
