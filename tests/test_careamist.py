import pytest

from careamics import CAREamist, Configuration


def test_no_parameters():
    with pytest.raises(ValueError):
        careamist = CAREamist()


def test_minimum_config(minimum_config):
    # create configuration
    config = Configuration(**minimum_config)

    # instantiate CAREamist
    CAREamist(config)