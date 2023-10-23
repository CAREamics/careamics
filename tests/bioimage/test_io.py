from pathlib import Path

import bioimageio.spec.shared.raw_nodes as nodes
import numpy as np
import torch
from bioimageio.core import load_resource_description

from careamics.config import Configuration, save_configuration
from careamics.engine import Engine
from careamics.models import create_model
from careamics.utils import cwd


def save_checkpoint(engine: Engine, config: Configuration) -> None:
    # create a fake checkpoint
    checkpoint = {
        "epoch": 1,
        "model_state_dict": engine.model.state_dict(),
        "optimizer_state_dict": engine.optimizer.state_dict(),
        "scheduler_state_dict": engine.lr_scheduler.state_dict(),
        "grad_scaler_state_dict": engine.scaler.state_dict(),
        "loss": 0.01,
        "config": config,
    }
    checkpoint_path = (
        Path(config["working_directory"])
        .joinpath(f"{config['experiment_name']}_best.pth")
        .absolute()
    )
    torch.save(checkpoint, checkpoint_path)


def test_bioimage_export_default(minimum_config: dict, tmp_path: Path, request):
    """Test model export to bioimage format by using default specs."""
    # create configuration and save it to disk
    minimum_config["data"]["mean"] = 666.666
    minimum_config["data"]["std"] = 42.420
    minimum_config["data"]["axes"] = "YX"

    config = Configuration(**minimum_config)
    config_file = tmp_path / "tmp_config.yml"
    save_configuration(config, config_file)

    # create an engine to export the model
    engine = Engine(config=config)

    # create a monkey patch for the input (array saved during first validation)
    engine._input = np.random.randint(0, 255, minimum_config["training"]["patch_size"])

    # save fake checkpoint
    save_checkpoint(engine, minimum_config)

    # output zip file
    zip_file = tmp_path / "tmp_model.bioimage.io.zip"

    # export the model (overriding the weight_uri)
    with cwd(tmp_path):
        engine.save_as_bioimage(zip_file)

        # put zip file in the cache for the import test
        request.config.cache.set("bioimage_model", str(zip_file))

        assert zip_file.exists()

        rdf = load_resource_description(zip_file)
        assert isinstance(rdf, nodes.ResourceDescription)


def test_bioimage_import(request):
    """Test model import from bioimage format.

    IMPORTANT: this test will only pass if `test_bioimage_export_default` passed.
    """
    zip_file = request.config.cache.get("bioimage_model", None)
    if zip_file is not None and Path(zip_file).exists():
        # checkpoint_path = import_bioimage_model(zip_file)
        _, _, _, _, config = create_model(model_path=zip_file)
        assert isinstance(config, Configuration)
    else:
        raise ValueError("No valid bioimage model zip provided.")
