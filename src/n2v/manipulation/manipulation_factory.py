from functools import partial

from typing import Callable

from ..config import Configuration
from . import pixel_manipulation


def create_patch_transform(config: Configuration) -> Callable:
    """Creates the patch transform function with optional augmentation
    Parameters
    ----------
    config : dict.

    Returns
    -------
    Callable
    """
    return partial(
        getattr(
            pixel_manipulation, f"{config.algorithm.pixel_manipulation}_manipulate"
        ),
        num_pixels=config.algorithm.num_masked_pixels,
        # TODO add augmentation selection
        augmentations=None,
    )
