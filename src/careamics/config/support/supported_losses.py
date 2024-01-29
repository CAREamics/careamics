from enum import Enum

class SupportedLoss(str, Enum):
    """
    Available loss functions.

    Currently supported losses:

        - n2v: Noise2Void loss.
    """

    MSE = "mse"
    MAE = "mae"
    N2V = "n2v"
    # PN2V = "pn2v"
    # HDN = "hdn"
    # CE = "ce"
    # DICE = "dice"
    # CUSTOM = "custom"