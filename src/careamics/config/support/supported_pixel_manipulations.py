"""Pixel manipulation methods supported by CAREamics."""

from enum import StrEnum


class SupportedPixelManipulation(StrEnum):
    """Supported Noise2Void pixel manipulations.

    - Uniform: Replace masked pixel value by a (uniformly) randomly selected neighbor
        pixel value.
    - Median: Replace masked pixel value by the mean of the neighborhood.
    """

    UNIFORM = "uniform"
    MEDIAN = "median"
