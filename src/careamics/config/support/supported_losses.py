from aenum import StrEnum


# TODO register loss with custom_loss decorator?
class SupportedLoss(StrEnum):
    #_init_ = 'value __doc__'

    MSE = "mse"
    MAE = "mae"
    N2V = "n2v"
    # PN2V = "pn2v"
    # HDN = "hdn"
    # CE = "ce"
    # DICE = "dice"
    # CUSTOM = "custom" # TODO create mechanism for that