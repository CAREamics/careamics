from careamics.losses.loss_factory import loss_factory
from careamics.losses.lvae.losses import denoisplit_loss, musplit_loss
import pytest


def test_mu_split_loss():
    loss = loss_factory("musplit")
    assert loss == musplit_loss


def test_denoisplit_loss():
    loss = loss_factory("denoisplit")
    assert loss == denoisplit_loss
