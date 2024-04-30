"""Transforms that are used to augment the data."""

__all__ = [
    "get_all_transforms",
    "N2VManipulate",
    "NDFlip",
    "XYRandomRotate90",
    "ImageRestorationTTA",
    "Denormalize",
    "Normalize",
]


from .n2v_manipulate import N2VManipulate
from .nd_flip import NDFlip
from .normalize import Denormalize, Normalize
from .tta import ImageRestorationTTA
from .xy_random_rotate90 import XYRandomRotate90

ALL_TRANSFORMS = {
    "Normalize": Normalize,
    "N2VManipulate": N2VManipulate,
    "NDFlip": NDFlip,
    "XYRandomRotate90": XYRandomRotate90,
}


def get_all_transforms() -> dict:
    """Return all the transforms accepted by CAREamics.

    Note that while CAREamics accepts any `Compose` transforms from Albumentations (see
    https://albumentations.ai/), only a few transformations are explicitely supported
    (see `SupportedTransform`).

    Returns
    -------
    dict
        A dictionary with all the transforms accepted by CAREamics, where the keys are
        the transform names and the values are the transform classes.
    """
    return ALL_TRANSFORMS
