"""Algorithms supported by CAREamics."""

from __future__ import annotations

from careamics.utils import BaseEnum


class SupportedAlgorithm(str, BaseEnum):
    """Algorithms available in CAREamics.

    # TODO
    """

    N2V = "n2v"
    """Noise2Void algorithm."""

    CARE = "care"
    """Content-aware image restoration algorithm."""

    N2N = "n2n"
    """Noise2Noise algorithm."""

    CUSTOM = "custom"
    """Custom algorithm, used for cases where a custom architecture is provided."""

    MUSPLIT = "musplit"
    DENOISPLIT = "denoisplit"
    # PN2V = "pn2v"
    # HDN = "hdn"
    # SEG = "segmentation"
