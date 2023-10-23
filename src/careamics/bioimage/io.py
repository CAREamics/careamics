"""Export to bioimage.io format."""
from pathlib import Path
from typing import Union

import numpy as np
import torch
from bioimageio.core import load_resource_description
from bioimageio.core.build_spec import build_model

from ..config.config import (
    Configuration,
)

PYTORCH_STATE_DICT = "pytorch_state_dict"


def save_bioimage_model(
    path: Union[str, Path],
    config: Configuration,
    model_specs: dict,
) -> None:
    """
    Build bioimage model zip file from model specification data.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the model zip file.
    config : Configuration
        Configuration object.
    model_specs : dict
        Model specification data.
    """
    workdir = config.working_directory

    # load best checkpoint
    checkpoint_path = workdir.joinpath(f"{config.experiment_name}_best.pth").absolute()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # save chekpoint entries in separate files
    weight_path = workdir.joinpath("model_weights.pth")
    torch.save(checkpoint["model_state_dict"], weight_path)

    optim_path = workdir.joinpath("optim.pth")
    torch.save(checkpoint["optimizer_state_dict"], optim_path)

    scheduler_path = workdir.joinpath("scheduler.pth")
    torch.save(checkpoint["scheduler_state_dict"], scheduler_path)

    grad_path = workdir.joinpath("grad.pth")
    torch.save(checkpoint["grad_scaler_state_dict"], grad_path)

    config_path = workdir.joinpath("config.pth")
    torch.save(config.model_dump(), config_path)

    # create cover
    # TODO: this is a hack to avoid an export error in bioimage.io when the
    # inputs/outputs have singleton dimensions.
    # load .npy files
    np.load(model_specs["test_inputs"][0]).squeeze()
    np.load(model_specs["test_outputs"][0]).squeeze()

    # create cover array

    # Create attachments
    attachments = [
        str(optim_path),
        str(scheduler_path),
        str(grad_path),
        str(config_path),
    ]

    model_specs.update(
        {
            "weight_type": PYTORCH_STATE_DICT,
            "weight_uri": str(weight_path),
            "attachments": {"files": attachments},
        }
    )

    if config.algorithm.is_3D:
        model_specs["tags"].append("3D")
    else:
        model_specs["tags"].append("2D")

    # build model zip
    build_model(
        output_path=Path(path).absolute(),
        **model_specs,
    )

    # remove the temporary files
    weight_path.unlink()
    optim_path.unlink()
    scheduler_path.unlink()
    grad_path.unlink()
    config_path.unlink()


def import_bioimage_model(model_path: Union[str, Path]) -> Path:
    """
    Load configuration and weights from a bioimage zip model.

    Parameters
    ----------
    model_path : Union[str, Path]
        Path to the bioimage.io archive.

    Returns
    -------
    Path
        Path to the checkpoint.

    Raises
    ------
    ValueError
        If the model format is invalid.
    FileNotFoundError
        If the checkpoint file was not found.
    """
    if isinstance(model_path, str):
        model_path = Path(model_path)
    # check the model extension (should be a zip file).
    if model_path.suffix != ".zip":
        raise ValueError("Invalid model format. Expected bioimage model zip file.")
    # load the model
    rdf = load_resource_description(model_path)

    # create a valid checkpoint file from weights and attached files
    basedir = model_path.parent.joinpath("rdf_model")
    basedir.mkdir(exist_ok=True)
    optim_path = None
    scheduler_path = None
    grad_path = None
    config_path = None
    weight_path = None

    if rdf.weights.get(PYTORCH_STATE_DICT) is not None:
        weight_path = rdf.weights.get(PYTORCH_STATE_DICT).source

    for file in rdf.attachments.files:
        if file.name.endswith("optim.pth"):
            optim_path = file
        elif file.name.endswith("scheduler.pth"):
            scheduler_path = file
        elif file.name.endswith("grad.pth"):
            grad_path = file
        elif file.name.endswith("config.pth"):
            config_path = file

    if (
        weight_path is None
        or optim_path is None
        or scheduler_path is None
        or grad_path is None
        or config_path is None
    ):
        raise FileNotFoundError(f"No valid checkpoint file was found in {model_path}.")

    checkpoint = {
        "model_state_dict": torch.load(weight_path, map_location="cpu"),
        "optimizer_state_dict": torch.load(optim_path, map_location="cpu"),
        "scheduler_state_dict": torch.load(scheduler_path, map_location="cpu"),
        "grad_scaler_state_dict": torch.load(grad_path, map_location="cpu"),
        "config": torch.load(config_path, map_location="cpu"),
    }
    checkpoint_path = basedir.joinpath("checkpoint.pth")
    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path
