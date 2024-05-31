"""Losses supported by CAREamics."""

from careamics.utils import BaseEnum


# TODO register loss with custom_loss decorator?
class SupportedLoss(str, BaseEnum):
    """Supported losses.

    Attributes
    ----------
    MSE : str
        Mean Squared Error loss.
    MAE : str
        Mean Absolute Error loss.
    N2V : str
        Noise2Void loss.
    """

    MSE = "mse"
    MAE = "mae"
    N2V = "n2v"
    # PN2V = "pn2v"
    # HDN = "hdn"
    # CE = "ce"
    # DICE = "dice"
    # CUSTOM = "custom" # TODO create mechanism for that
