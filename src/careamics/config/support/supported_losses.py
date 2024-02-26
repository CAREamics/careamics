from careamics.utils import BaseEnum


# TODO register loss with custom_loss decorator?
class SupportedLoss(str, BaseEnum):

    MSE = "mse"
    MAE = "mae"
    N2V = "n2v"
    # PN2V = "pn2v"
    # HDN = "hdn"
    # CE = "ce"
    # DICE = "dice"
    # CUSTOM = "custom" # TODO create mechanism for that