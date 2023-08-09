from pathlib import Path

import bioimageio.spec.shared.raw_nodes as nodes
import pytest
import torch
from bioimageio.core import load_resource_description

from careamics_restoration.bioimage.io import get_default_model_specs
from careamics_restoration.config import Configuration, save_configuration
from careamics_restoration.engine import Engine
from careamics_restoration.models import create_model


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


@pytest.mark.parametrize("name", ["Noise2Void"])
@pytest.mark.parametrize("is_3D", [True, False])
def test_default_model_specs(name, is_3D):
    mean = 666.666
    std = 42.420

    if is_3D:
        axes = "zyx"
    else:
        axes = "yx"

    specs = get_default_model_specs(name, mean, std, is_3D=is_3D)
    assert specs["name"] == name
    assert specs["preprocessing"][0][0]["kwargs"]["mean"] == [mean]
    assert specs["preprocessing"][0][0]["kwargs"]["std"] == [std]
    assert specs["preprocessing"][0][0]["kwargs"]["axes"] == axes
    assert specs["postprocessing"][0][0]["kwargs"]["offset"] == [mean]
    assert specs["postprocessing"][0][0]["kwargs"]["gain"] == [std]
    assert specs["postprocessing"][0][0]["kwargs"]["axes"] == axes


def test_bioimage_export_default(minimum_config: dict, tmp_path: Path, request):
    """Test model export to bioimage format by using default specs."""

    # create configuration and save it to disk
    minimum_config["data"]["mean"] = 666.666
    minimum_config["data"]["std"] = 42.420

    config = Configuration(**minimum_config)
    config_file = tmp_path / "tmp_config.yml"
    save_configuration(config, config_file)

    # create an engine to export the model
    engine = Engine(config=config)

    # save fake checkpoint
    save_checkpoint(engine, minimum_config)

    # output zip file
    zip_file = tmp_path.joinpath("tmp_model.zip")

    # export the model (overriding the weight_uri)
    engine.save_as_bioimage(zip_file)

    # put zip file in the cache for the import test
    request.config.cache.set("bioimage_model", str(zip_file))

    assert zip_file.exists()

    rdf = load_resource_description(zip_file)
    assert isinstance(rdf, nodes.ResourceDescription)


def test_bioimage_export_without_mean_std(minimum_config: dict, tmp_path: Path):
    """Test that model export to bioimage format without specifying mean and std
    raises an error."""

    # create configuration and save it to disk
    config = Configuration(**minimum_config)
    config_file = tmp_path / "tmp_config.yml"
    save_configuration(config, config_file)

    # create an engine to export the model
    engine = Engine(config=config)

    # save fake checkpoint
    save_checkpoint(engine, minimum_config)

    # output zip file
    zip_file = tmp_path.joinpath("tmp_model.zip")

    with pytest.raises(ValueError):
        engine.save_as_bioimage(zip_file)

    # test if error is raised when config is None
    engine.config = None

    with pytest.raises(ValueError):
        engine.save_as_bioimage(zip_file)


def test_bioimage_import(request):
    zip_file = request.config.cache.get("bioimage_model", None)
    if zip_file is not None and Path(zip_file).exists():
        # checkpoint_path = import_bioimage_model(zip_file)
        _, _, _, _, config = create_model(model_path=zip_file)
        assert isinstance(config, Configuration)
    else:
        raise ValueError("No valid bioimage model zip provided.")
