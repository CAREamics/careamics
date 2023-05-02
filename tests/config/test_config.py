from pathlib import Path

import pytest
import yaml

from n2v.config import Configuration

# TODO test optimizer and lr schedulers parameters


def export_to_yaml(folder: Path, config: Configuration) -> Path:
    yaml_path = Path(folder, "data.yml")
    with open(yaml_path, "w") as f:
        yaml.dump(config.dict(), f, default_flow_style=False)
    assert yaml_path.exists()

    return yaml_path


def test_config_to_yaml(tmpdir, test_config):
    """Test that we can export a config to yaml and load it back"""

    # test that we can instantiate a config
    myconf = Configuration(**test_config)

    # export to yaml
    yaml_path = export_to_yaml(tmpdir, myconf)

    # load from yaml
    with open(yaml_path) as f:
        config_yaml = yaml.safe_load(f)

        # parse yaml
        my_other_conf = Configuration(**config_yaml)

        assert my_other_conf == myconf


def test_config_optional(tmpdir, test_config):
    """Test that we can export a partial config to yaml and load it back.

    In this case a partial config has none of the optional fields (training,
    evaluation, prediction).
    """
    # remove optional entries
    del test_config["training"]
    del test_config["evaluation"]
    del test_config["prediction"]

    # instantiate configuration
    myconf = Configuration(**test_config)

    # export to yaml
    yaml_path = export_to_yaml(tmpdir, myconf)

    # load from yaml
    with open(yaml_path) as f:
        config_yaml = yaml.safe_load(f)

        # parse yaml
        my_other_conf = Configuration(**config_yaml)

        assert my_other_conf == myconf


def test_config_non_existing_workdir(test_config):
    """Test that we cannot instantiate a config with non existing workdir."""

    config = test_config
    config["workdir"] = "non_existing_workdir"

    with pytest.raises(ValueError):
        Configuration(**config)


def test_config_stage(test_config):
    """Test that we can get the configuration for a specific stage."""

    # test that we can instantiate a config
    myconf = Configuration(**test_config)

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
