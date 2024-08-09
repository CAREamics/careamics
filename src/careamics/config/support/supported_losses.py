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

    MSE = "mse_loss"
    MAE = "mae_loss"
    N2V = "n2v_loss"
    # PN2V = "pn2v"
    # HDN = "hdn"
    MUSPLIT = "musplit_loss"
    DENOISPLIT = "denoisplit_loss"
    # CE = "ce"
    # DICE = "dice"
