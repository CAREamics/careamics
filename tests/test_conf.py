from pathlib import Path

import pytest
import yaml

from n2v.config import Config
from n2v.config.config import Algorithm


# TODO test optimizer and lr schedulers parameters


def test_config_enum_wrong_values():
    """Test that we can't instantiate a config with wrong enum values"""

    mydict = {
        "name": "myalgo",
        "loss": ["notn2v"],
        "model": "UNet",
        "num_masked_pixels": 128,
        "patch_size": [64, 64],
    }

    with pytest.raises(ValueError):
        Algorithm(**mydict)


def test_config_to_yaml(tmpdir, test_config):
    """Test that we can export a config to yaml and load it back"""

    # test that we can instantiate a config
    myconf = Config(**test_config)

    # export to yaml
    yaml_path = Path(tmpdir, "data.yml")
    with open(yaml_path, "w") as f:
        yaml.dump(myconf.dict(), f, default_flow_style=False)
    assert yaml_path.exists()

    # load from yaml
    with open(yaml_path) as f:
        config_yaml = yaml.safe_load(f)

        # parse yaml
        my_other_conf = Config(**config_yaml)

        assert my_other_conf == myconf

    # TODO remove the following code (this is just for demo):
    assert "n2v" in myconf.algorithm.loss
