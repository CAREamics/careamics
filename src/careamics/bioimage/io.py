"""Export to bioimage.io format."""

from pathlib import Path
from typing import Union

import torch
from bioimageio.core import load_resource_description
from bioimageio.core.build_spec import build_model

from careamics.config.config import Configuration
from careamics.utils.context import cwd

PYTORCH_STATE_DICT = "pytorch_state_dict"


def save_bioimage_model(
    path: Union[str, Path],
    config: Configuration,
    specs: dict,
) -> None:
    """
    Build bioimage model zip file from model RDF data.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the model zip file.
    config : Configuration
        Configuration object.
    specs : dict
        Model RDF dict.
    """
    workdir = config.working_directory

    # temporary folder
    temp_folder = Path.home().joinpath(".careamics", "bmz_tmp")
    temp_folder.mkdir(exist_ok=True, parents=True)

    # change working directory to the temp folder
    with cwd(temp_folder):
        # load best checkpoint
        checkpoint_path = workdir.joinpath(
            f"{config.experiment_name}_best.pth"
        ).absolute()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # save chekpoint entries in separate files
        weight_path = Path("model_weights.pth")
        torch.save(checkpoint["model_state_dict"], weight_path)

        optim_path = Path("optim.pth")
        torch.save(checkpoint["optimizer_state_dict"], optim_path)

        scheduler_path = Path("scheduler.pth")
        torch.save(checkpoint["scheduler_state_dict"], scheduler_path)

        grad_path = Path("grad.pth")
        torch.save(checkpoint["grad_scaler_state_dict"], grad_path)

        config_path = Path("config.pth")
        torch.save(config.model_dump(), config_path)

        # create attachments
        attachments = [
            str(optim_path),
            str(scheduler_path),
            str(grad_path),
            str(config_path),
        ]

        # create requirements file
        requirements = Path("requirements.txt")
        with open(requirements, "w") as f:
            f.write("git+https://github.com/CAREamics/careamics.git")

        algo_config = config.algorithm
        specs.update(
            {
                "weight_type": PYTORCH_STATE_DICT,
                "weight_uri": str(weight_path),
                "architecture": "careamics.models.unet.UNet",
                "pytorch_version": torch.__version__,
                "model_kwargs": {
                    "conv_dim": algo_config.get_conv_dim(),
                    "depth": algo_config.model_parameters.depth,
                    "num_channels_init": algo_config.model_parameters.num_channels_init,
                },
                "dependencies": "pip:" + str(requirements),
                "attachments": {"files": attachments},
            }
        )

        if config.algorithm.is_3D:
            specs["tags"].append("3D")
        else:
            specs["tags"].append("2D")

        # build model zip
        build_model(
            output_path=Path(path).absolute(),
            **specs,
        )

        # remove temporary files
        for file in temp_folder.glob("*"):
            file.unlink()

    # delete temporary folder
    temp_folder.rmdir()


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
