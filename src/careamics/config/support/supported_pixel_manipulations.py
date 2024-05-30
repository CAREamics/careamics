"""Pixel manipulation methods supported by CAREamics."""

from careamics.utils import BaseEnum


class SupportedPixelManipulation(str, BaseEnum):
    """Supported Noise2Void pixel manipulations.

    - Uniform: Replace masked pixel value by a (uniformly) randomly selected neighbor
        pixel value.
    - Median: Replace masked pixel value by the mean of the neighborhood.
    """

    UNIFORM = "uniform"
    MEDIAN = "median"
