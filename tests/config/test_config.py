import pytest

from careamics_restoration.config import (
    Configuration,
    load_configuration,
    save_configuration,
)


def test_minimum_config(minimum_config):
    """Test that we can instantiate a minimum config."""
    myconf = Configuration(**minimum_config)
    dictionary = myconf.dict()

    assert dictionary["algorithm"] == minimum_config["algorithm"]
    assert dictionary["data"] == minimum_config["data"]
    assert dictionary["training"] == minimum_config["training"]
    assert myconf.dict() == minimum_config


def test_config_to_yaml(tmp_path, minimum_config):
    """Test that we can export a config to yaml and load it back"""

    # test that we can instantiate a config
    myconf = Configuration(**minimum_config)

    # export to yaml
    yaml_path = save_configuration(myconf, tmp_path)
    assert yaml_path.exists()

    # load from yaml
    config_yaml = load_configuration(yaml_path)

    # parse yaml
    my_other_conf = Configuration(**config_yaml)

    assert my_other_conf == myconf


def test_get_stage(complete_config):
    """Test that we can get the configuration for a specific stage."""

    # test that we can instantiate a config
    myconf = Configuration(**complete_config)

    # get training config
    training_config = myconf.get_stage_config("training")
    assert training_config == myconf.training

    # get prediction config
    prediction_config = myconf.get_stage_config("prediction")
    assert prediction_config == myconf.prediction

    with pytest.raises(ValueError):
        myconf.get_stage_config("not_a_stage")
