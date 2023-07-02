from functools import partial
from typing import Callable

from ..config import Configuration
from . import pixel_manipulation


# TODO this function is too complex for such a simple task
def create_masking_transform(config: Configuration) -> Callable:
    """Creates the patch transform function with optional augmentation
    Parameters
    ----------
    config : dict.

    Returns
    -------
    Callable
    """
    return partial(
        getattr(pixel_manipulation, f"{config.algorithm.masking_strategy}_manipulate"),
        mask_pixel_percentage=config.algorithm.masked_pixel_percentage,
        # TODO add augmentation selection
        augmentations=None,
    )
