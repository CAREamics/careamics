"""Algorithms supported by CAREamics."""

from __future__ import annotations

from enum import StrEnum


class SupportedAlgorithm(StrEnum):
    """Algorithms available in CAREamics.

    These definitions are the same as the keyword `name` of the algorithm
    configurations.
    """

    # --- unet-based algorithms
    N2V = "n2v"
    """Noise2Void algorithm, a self-supervised approach based on blind denoising."""

    CARE = "care"
    """Content-aware image restoration, a supervised algorithm used for a variety
    of tasks."""

    N2N = "n2n"
    """Noise2Noise algorithm, a self-supervised denoising scheme based on comparing
    noisy images of the same sample."""

    PN2V = "pn2v"
    """Probabilistic Noise2Void. A extension of Noise2Void using noise models."""

    SEG = "seg"
    """UNet-based semantic segmentation."""

    # --- vae-based algorithms
    MICROSPLIT = "microsplit"
    """An image splitting and denoising approach based on ladder VAE architectures."""

    HDN = "hdn"
    """Hierarchical DivNoising, an unsupervised denoising algorithm capable of removing
    structured noise."""
