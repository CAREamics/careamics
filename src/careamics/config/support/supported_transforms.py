from inspect import getmembers, isclass

from aenum import StrEnum
import albumentations as Aug

from  careamics import transforms


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


class SupportedTransform(StrEnum):
    """Transforms officially supported by CAREamics.

    - Flip, from Albumentations
    - RandomRotate90, from Albumentations
    - NormalizeWithoutTarget # TODO add details, in particular about the parameters
    - ManipulateN2V # TODO add details, in particular about the parameters

    Note that while any Albumentations (see https://albumentations.ai/) transform can be
    used in CAREamics, no check are implemented to verify the compatibility of any other
    transforms than the ones officially supported. 
    """
    _init_ = 'value __doc__'

    FLIP = "Flip", "Randomly flip the input horizontally, vertically or both, "\
        "parameter `p` can be used to set the probability to apply the transform."
    RANDOM_ROTATE90 = "RandomRotate90", "Randomly rotate the input by 90 degrees, "\
        "parameter `p` can be used to set the probability to apply the transform."
    NORMALIZE_WO_TARGET = "NormalizeWithoutTarget", "" # TODO add docstring
    MANIPULATE_N2V = "ManipulateN2V", "" # TODO add docstring
    # CUSTOM = "Custom"