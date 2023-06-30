import pytest

from careamics_restoration.config import (
    Configuration,
    load_configuration,
    save_configuration,
)

# TODO test optimizer and lr schedulers parameters


def test_minimum_config(minimum_config):
    """Test that we can instantiate a minimum config."""
    myconf = Configuration(**minimum_config)

    assert myconf.experiment_name == minimum_config["experiment_name"]
    assert myconf.working_directory == minimum_config["working_directory"]


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


def test_optional_entries(tmp_path, minimum_config):
    """Test that we can export a partial config to yaml and load it back.

    In this case a partial config has none of the optional fields (training,
    evaluation, prediction).
    """
    # remove optional entries
    del minimum_config["training"]
    del minimum_config["evaluation"]
    del minimum_config["prediction"]

    # instantiate configuration
    myconf = Configuration(**minimum_config)

    # export to yaml
    yaml_path = save_configuration(myconf, tmp_path)
    assert yaml_path.exists()

    # load from yaml
    config_yaml = load_configuration(yaml_path)

    # parse yaml
    my_other_conf = Configuration(**config_yaml)
    assert my_other_conf == myconf


def test_get_stage(minimum_config):
    """Test that we can get the configuration for a specific stage."""

    # test that we can instantiate a config
    myconf = Configuration(**minimum_config)

    # get training config
    training_config = myconf.get_stage_config("training")
    assert training_config == myconf.training

    # get evaluation config
    evaluation_config = myconf.get_stage_config("evaluation")
    assert evaluation_config == myconf.evaluation

    # get prediction config
    prediction_config = myconf.get_stage_config("prediction")
    assert prediction_config == myconf.prediction

    with pytest.raises(ValueError):
        myconf.get_stage_config("not_a_stage")
