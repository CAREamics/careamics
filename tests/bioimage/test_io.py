from pathlib import Path

import yaml
import pytest

from careamics_restoration.engine import Engine
from careamics_restoration.config import Configuration
from careamics_restoration.bioimage import import_bioimage_model


def test_bioimage_export_default(minimum_config: dict, tmp_path: Path, request):
    """Test model export to bioimage forrmat by using default specs."""
    # dump configs to a tmp file:
    config_file = tmp_path.joinpath("tmp_config.yml")
    with open(config_file, "w") as f:
        yaml.safe_dump(minimum_config, f)
    # create an engine to export the model:
    engine = Engine(config_file)
    # save a tmp model's weights file
    weight_file = tmp_path.joinpath(f"{minimum_config['experiment_name']}_best.pth")
    with open(weight_file, "wb") as f:
        f.write(b"model weights!")
    # output zip file
    zip_file = tmp_path.joinpath("tmp_model.zip")
    # export the model (overriding the weight_uri)
    engine.save_as_bioimage(zip_file, {"weight_uri": str(weight_file)})

    # put zip file in the cache for the import test
    request.config.cache.set("bioimage_model", str(zip_file))

    assert zip_file.exists()


def test_bioimage_import(request):
    zip_file = request.config.cache.get("bioimage_model", None)
    if zip_file is not None and Path(zip_file).exists():
        config = import_bioimage_model(zip_file)
        assert isinstance(config, Configuration)
    else:
        raise ValueError("No bioimage model zip provided.")
