from inspect import getmembers, isclass

import albumentations as Aug

from careamics import transforms
from careamics.utils import BaseEnum

ALL_TRANSFORMS = dict(getmembers(Aug, isclass) + getmembers(transforms, isclass))


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


class SupportedTransform(str, BaseEnum):
    """Transforms officially supported by CAREamics.

    - Flip: from Albumentations, randomly flip the input horizontally, vertically or
        both, parameter `p` can be used to set the probability to apply the transform.
    - XYRandomRotate90: #TODO
    - Normalize # TODO add details, in particular about the parameters
    - ManipulateN2V # TODO add details, in particular about the parameters
    - NDFlip

    Note that while any Albumentations (see https://albumentations.ai/) transform can be
    used in CAREamics, no check are implemented to verify the compatibility of any other
    transforms than the ones officially supported.
    """

    NDFLIP = "NDFlip"
    XY_RANDOM_ROTATE90 = "XYRandomRotate90"
    NORMALIZE = "Normalize"
    N2V_MANIPULATE = "N2VManipulate"
    # CUSTOM = "Custom"
