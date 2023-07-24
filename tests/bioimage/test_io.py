from pathlib import Path

import torch
import yaml

from careamics_restoration.bioimage import import_bioimage_model
from careamics_restoration.config import Configuration
from careamics_restoration.engine import Engine


def test_bioimage_export_default(minimum_config: dict, tmp_path: Path, request):
    """Test model export to bioimage forrmat by using default specs."""
    # dump configs to a tmp file:
    config_file = tmp_path.joinpath("tmp_config.yml")
    with open(config_file, "w") as f:
        yaml.safe_dump(minimum_config, f)
    # create an engine to export the model:
    engine = Engine(config_path=config_file)
    # create a fake checkpoint!
    checkpoint = {
        "epoch": 1,
        "model_state_dict": {"weight_1": [0.0, 0.1, 0.001]},
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
        "grad_scaler_state_dict": {},
        "loss": 0.01,
        "config": {},
    }
    checkpoint_path = (
        Path(minimum_config["working_directory"])
        .joinpath(f"{minimum_config['experiment_name']}_best.pth")
        .absolute()
    )
    torch.save(checkpoint, checkpoint_path)
    # output zip file
    zip_file = tmp_path.joinpath("tmp_model.zip")
    # export the model (overriding the weight_uri)
    engine.save_as_bioimage(zip_file)

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
