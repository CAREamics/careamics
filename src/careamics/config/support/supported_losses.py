"""Losses supported by CAREamics."""

from enum import StrEnum


# TODO register loss with custom_loss decorator?
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
    HDN : str
        Hierarchical DivNoising loss.
    MICROSPLIT : str
        MicroSplit loss.
    """

    MSE = "mse"
    MAE = "mae"
    N2V = "n2v"
    PN2V = "pn2v"
    HDN = "hdn"
    MICROSPLIT = "microsplit"
