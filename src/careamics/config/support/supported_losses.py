"""Losses supported by CAREamics."""

from enum import StrEnum


class SupportedLoss(StrEnum):
    """Supported losses.

    Attributes
    ----------
    MSE : str
        Mean Squared Error loss.
    MAE : str
        Mean Absolute Error loss.
    N2V : str
        Noise2Void loss.
    PN2V : str
        Probabilistic Noise2Void loss.
    CE : str
        Cross-Entropy loss.
    DICE : str
        Dice loss.
    DICE_CE : str
        Combined Dice and Cross-Entropy loss.
    HDN : str
        Hierarchical DivNoising loss.
    """

    # --- CARE and Noise2Noise losses
    MSE = "mse"
    MAE = "mae"

    # --- Noise2Void losses
    N2V = "n2v"
    PN2V = "pn2v"

    # --- Segmentation losses
    CE = "ce"
    DICE = "dice"
    DICE_CE = "dice_ce"

    # --- VAE losses
    HDN = "hdn"
    MUSPLIT = "musplit"
    MICROSPLIT = "microsplit"
    DENOISPLIT = "denoisplit"
    DENOISPLIT_MUSPLIT = (
        "denoisplit_musplit"  # TODO refac losses, leave only microsplit
    )
