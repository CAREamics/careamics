from pathlib import Path

import yaml

from n2v.config import Config


def test_config_to_yaml(tmpdir):
    """Test that we can export a config to yaml and load it back"""
    myconf_dict = {
        "experiment_name": "testing",
        "workdir": str(tmpdir),
        "algorithm": {
            "name": "myalgo",
            "loss": ["n2v", "pn2v"],
            "model": "UNet",
            "num_masked_pixels": 128,
            "patch_size": [64, 64],
        },
    }

    # test that we can instantiate a config
    myconf = Config(**myconf_dict)

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
