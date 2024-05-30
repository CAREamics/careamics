"""Algorithms supported by CAREamics."""

from __future__ import annotations

from careamics.utils import BaseEnum


class SupportedAlgorithm(str, BaseEnum):
    """Algorithms available in CAREamics.

    # TODO
    """

    N2V = "n2v"
    CARE = "care"
    N2N = "n2n"
    CUSTOM = "custom"
    # PN2V = "pn2v"
    # HDN = "hdn"
    # SEG = "segmentation"
