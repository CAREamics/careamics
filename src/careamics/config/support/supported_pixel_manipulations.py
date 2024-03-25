from careamics.utils import BaseEnum


class SupportedPixelManipulation(str, BaseEnum):
    """_summary_.

    - Uniform: Replace masked pixel value by a (uniformly) randomly selected neighbor
        pixel value.
    - Median: Replace masked pixel value by the mean of the neighborhood.
    """

    # TODO docs

    UNIFORM = "uniform"
    MEDIAN = "median"
