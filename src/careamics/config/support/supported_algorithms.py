"""Algorithms supported by CAREamics."""

from __future__ import annotations

from careamics.utils import BaseEnum


class SupportedAlgorithm(str, BaseEnum):
    """Algorithms available in CAREamics.

    These definitions are the same as the keyword `name` of the algorithm
    configurations.
    """

    # UNet-based algorithms
    N2V = "n2v"
    """Noise2Void algorithm, a self-supervised approach based on blind denoising."""

    PN2V = "pn2v"
    """Probabilistic Noise2Void. A extension of Noise2Void is not restricted to Gaussian
    noise models or Gaussian intensity predictions."""

    CARE = "care"
    """Content-aware image restoration, a supervised algorithm used for a variety
    of tasks."""

    N2N = "n2n"
    """Noise2Noise algorithm, a self-supervised denoising scheme based on comparing
    noisy images of the same sample."""

    SEG = "seg"
    """Segmentation algorithm based on UNet architecture."""

    # VAE-based algorithms
    HDN = "hdn"
    """Hierarchical DivNoising, an unsupervised denoising algorithm."""

    MICROSPLIT = "microsplit"
    """A micro-level image splitting approach based on ladder VAE architectures."""
