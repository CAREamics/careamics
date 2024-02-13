from enum import Enum


# TODO register loss with custom_loss decorator?
class SupportedLoss(str, Enum):

    MSE = "mse"
    MAE = "mae"
    N2V = "n2v"
    # PN2V = "pn2v"
    # HDN = "hdn"
    # CE = "ce"
    # DICE = "dice"
    # CUSTOM = "custom" # TODO create mechanism for that