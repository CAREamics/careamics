"""Losses supported by CAREamics."""

from careamics.utils import BaseEnum


# TODO register loss with custom_loss decorator?
class SupportedLoss(str, BaseEnum):
    """Supported losses."""

    # denoising losses
    MSE = "mse"
    MAE = "mae"

    # n2v family specific losses
    N2V = "n2v"
    PN2V = "pn2v"

    # lvae losses
    HDN = "hdn"
    MUSPLIT = "musplit"
    MICROSPLIT = "microsplit"
    DENOISPLIT = "denoisplit"
    DENOISPLIT_MUSPLIT = (
        "denoisplit_musplit"  # TODO refac losses, leave only microsplit
    )

    # segmentation losses
    CE = "ce"
    DICE = "dice"
    DICE_CE = "dice_ce"
