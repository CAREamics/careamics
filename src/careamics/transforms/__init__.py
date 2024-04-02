"""Transforms that are used to augment the data."""


from inspect import getmembers, isclass

import albumentations as Aug

from .n2v_manipulate import N2VManipulate
from .nd_flip import NDFlip
from .tta import ImageRestorationTTA
from .xy_random_rotate90 import XYRandomRotate90

ALL_TRANSFORMS = dict(getmembers(Aug, isclass) + [
    ("N2VManipulate", N2VManipulate),
    ("NDFlip", NDFlip),
    ("XYRandomRotate90", XYRandomRotate90),
])


def get_all_transforms() -> dict:
    """Return all the transforms accepted by CAREamics.

    This includes all transforms from Albumentations (see https://albumentations.ai/),
    and custom transforms implemented in CAREamics.

    Note that while any Albumentations transform can be used in CAREamics, no check are
    implemented to verify the compatibility of any other transforms than the ones
    officially supported (see SupportedTransforms).

    Returns
    -------
    dict
        A dictionary with all the transforms accepted by CAREamics, where the keys are
        the transform names and the values are the transform classes.
    """
    return ALL_TRANSFORMS
