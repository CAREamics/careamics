from pathlib import Path

import yaml
import pytest

from careamics_restoration.engine import Engine


def test_bioimage_export_default(minimum_config: dict, tmp_path: Path):
    """Test model export to bioimage forrmat by using default specs."""
    # dump configs to a tmp file.
    config_file = tmp_path.joinpath("tmp_config.yml")
    with open(config_file, 'w') as f:
        yaml.safe_dump(minimum_config, f)
    # create an engine to export
    engine = Engine(config_file)
    zip_file = tmp_path.joinpath("tmp_model.zip")  # model zip file
    engine.save_as_bioimage(zip_file)

    assert zip_file.exists()
