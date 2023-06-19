import pytest

from careamics_restoration.config.training import Optimizer


def test_optimizer_enum(test_config):
    optim_dict = test_config["training"]["optimizer"]

    opt = Optimizer(**optim_dict)

    assert opt.dict()["name"] == optim_dict["name"]
