from aenum import StrEnum


class SupportedPixelManipulation(StrEnum):
    # TODO docs
    _init_ = 'value __doc__'

    UNIFORM = "Uniform", "Replace masked pixel value by a (uniformly) randomly "\
        "selected neighbor pixel value."
    MEDIAN = "Median", "Replace masked pixel value by the mean of the neighborhood."