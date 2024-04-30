from careamics.utils import BaseEnum


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
