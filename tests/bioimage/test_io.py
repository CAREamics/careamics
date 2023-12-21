from pathlib import Path

import numpy as np
import pytest
import torch
from bioimageio.core import resource_tests

from careamics.bioimage import import_bioimage_model
from careamics.config import Configuration
from careamics.engine import Engine
from careamics.models import create_model


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


@pytest.mark.parametrize(
    "axes, patch",
    [
        ("YX", [64, 64]),
        ("ZYX", [32, 64, 64]),
    ],
)
def test_bioimage_io(minimum_config: dict, tmp_path: Path, axes, patch):
    """Test model export/import to bioimage format."""
    # create configuration
    minimum_config["data"]["mean"] = 666.666
    minimum_config["data"]["std"] = 42.420
    minimum_config["data"]["axes"] = axes
    minimum_config["training"]["patch_size"] = patch
    minimum_config["algorithm"]["is_3D"] = len(axes) == 3

    config = Configuration(**minimum_config)

    # create an engine to export the model
    engine = Engine(config=config)

    # create a monkey patch for the input (array saved during first validation)
    engine._input = np.random.randint(0, 255, minimum_config["training"]["patch_size"])
    engine._input = engine._input[np.newaxis, np.newaxis, ...]

    # save fake checkpoint
    save_checkpoint(engine, minimum_config)

    # output zip file
    zip_file = tmp_path / "tmp_model.bioimage.io.zip"

    engine.save_as_bioimage(zip_file.absolute())
    assert zip_file.exists()

    # load model
    _, _, _, _, loaded_config = create_model(model_path=zip_file)
    assert isinstance(loaded_config, Configuration)

    # check that the configuration is the same
    assert loaded_config == config

    # validate model
    results = resource_tests.test_model(zip_file)
    for result in results:
        assert result["status"] == "passed", f"Failed at {result['name']}."


def test_bioimage_wrong_path(tmp_path: Path):
    """Test that the model export fails if the path is wrong."""
    path = tmp_path / "wrong_path.tiff"
    with pytest.raises(ValueError):
        import_bioimage_model(path)
