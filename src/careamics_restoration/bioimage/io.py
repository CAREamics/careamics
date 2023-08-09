from pathlib import Path
from typing import Union

import torch
from bioimageio.core import load_resource_description
from bioimageio.core.build_spec import build_model
from bioimageio.spec.model.raw_nodes import Model

from careamics_restoration.config.config import (
    Configuration,
)

PYTORCH_STATE_DICT = "pytorch_state_dict"


def _get_model_doc(name: str) -> str:
    """Return markdown documentation path for a given model."""
    doc = Path(__file__).parent.joinpath("docs").joinpath(f"{name}.md")
    if doc.exists():
        return str(doc.absolute())
    else:
        raise FileNotFoundError(f"Documentation for {name} was not found.")


def get_default_model_specs(
    name: str, mean: float, std: float, is_3D: bool = False
) -> dict:
    """Return the default bioimage.io specs for given model's name.

    Currently only supports `n2v` model.

    Parameters
    ----------
    name : str
        Algorithm's name.
    mean : float
        Mean of the dataset.
    std : float
        Std of the dataset.
    is_3D : bool, optional
        Whether the model is 3D or not, by default False.

    Returns
    -------
    dict
        Model specs compatible with bioimage.io export.
    """
    rdf = {
        "name": "Noise2Void",
        "description": "Self-supervised denoising.",
        "license": "BSD-3-Clause",
        "authors": [
            {"name": "Alexander Krull"},
            {"name": "Tim-Oliver Buchholz"},
            {"name": "Florian Jug"},
        ],
        "cite": [
            {
                "doi": "10.48550/arXiv.1811.10980",
                "text": 'A. Krull, T.-O. Buchholz and F. Jug, "Noise2Void - Learning '
                'Denoising From Single Noisy Images," 2019 IEEE/CVF '
                "Conference on Computer Vision and Pattern Recognition "
                "(CVPR), 2019, pp. 2124-2132",
            }
        ],
        # "input_axes": ["bcyx"], <- overriden in save_as_bioimage
        "preprocessing": [  # for multiple inputs
            [  # multiple processes per input
                {
                    "kwargs": {
                        "axes": "zyx" if is_3D else "yx",
                        "mean": [mean],
                        "mode": "fixed",
                        "std": [std],
                    },
                    "name": "zero_mean_unit_variance",
                }
            ]
        ],
        # "output_axes": ["bcyx"], <- overriden in save_as_bioimage
        "postprocessing": [  # for multiple outputs
            [  # multiple processes per input
                {
                    "kwargs": {
                        "axes": "zyx" if is_3D else "yx",
                        "gain": [std],
                        "offset": [mean],
                    },
                    "name": "scale_linear",
                }
            ]
        ],
        "tags": ["unet", "denoising", "Noise2Void", "tensorflow", "napari"],
    }

    rdf["documentation"] = _get_model_doc(name)

    return rdf


def build_zip_model(
    path: Union[str, Path],
    config: Configuration,
    model_specs: dict,
) -> Model:
    """Build bioimage model zip file from model specification data.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the model zip file.
    config : Configuration
        Configuration object.
    model_specs : dict
        Model specification data.

    Returns
    -------
    Model
        Bioimage model object.
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
    raw_model = build_model(
        output_path=Path(path).absolute(),
        **model_specs,
    )

    # remove the temporary files
    weight_path.unlink()
    optim_path.unlink()
    scheduler_path.unlink()
    grad_path.unlink()
    config_path.unlink()

    return raw_model


def import_bioimage_model(model_path: Union[str, Path]) -> Path:
    """Load configs and weights from a bioimage zip model.

    Parameters
    ----------
        model_path (Union[str, Path]): Path to the bioimage model

    Return
    ------
        Path to the model's checkpoint file
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
