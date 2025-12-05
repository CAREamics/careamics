from pathlib import Path

import pytest

from careamics.config import (
    Configuration,
    load_configuration,
    save_configuration,
)


def test_config_to_yaml(tmp_path: Path, minimum_supervised_configuration: dict):
    """Test that we can export a config to yaml and load it back"""

    # test that we can instantiate a config
    myconf = Configuration(**minimum_supervised_configuration)

    # export to yaml
    yaml_path = save_configuration(myconf, tmp_path)
    assert yaml_path.exists()

    # load from yaml
    my_other_conf = load_configuration(yaml_path)
    assert my_other_conf == myconf


def test_config_to_yaml_wrong_path(
    tmp_path: Path, minimum_supervised_configuration: dict
):
    """Test that an error is raised when the path is not a directory and not a .yml"""

    # test that we can instantiate a config
    myconf = Configuration(**minimum_supervised_configuration)

    # export to yaml
    yaml_path = tmp_path / "tmp.txt"
    with pytest.raises(ValueError):
        save_configuration(myconf, yaml_path)

    # existing file
    yaml_path.touch()
    with pytest.raises(ValueError):
        save_configuration(myconf, yaml_path)
