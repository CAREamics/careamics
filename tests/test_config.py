from pathlib import Path

import pytest
import yaml

from n2v.config import Configuration
from n2v.config.config import Algorithm

# TODO test optimizer and lr schedulers parameters


def export_to_yaml(folder: Path, config: Configuration) -> Path:
    yaml_path = Path(folder, "data.yml")
    with open(yaml_path, "w") as f:
        yaml.dump(config.dict(), f, default_flow_style=False)
    assert yaml_path.exists()

    return yaml_path


def test_config_enum_wrong_values(test_config):
    """Test that we can't instantiate a config with wrong enum values"""

    algorithm_config = test_config["algorithm"]
    algorithm_config["loss"] = ["notn2v"]

    with pytest.raises(ValueError):
        Algorithm(**algorithm_config)


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
