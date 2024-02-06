from enum import Enum


# TODO: custom
class SupportedTransform(str, Enum):

    FLIP = "Flip"
    RANDOM_ROTATE90 = "RandomRotate90"
    NORMALIZE_WO_TARGET = "NormalizeWithoutTarget"
    MANIPULATE_N2V = "ManipulateN2V"
    CUSTOM = "Custom"