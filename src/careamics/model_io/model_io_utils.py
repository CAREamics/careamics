"""Utility functions to load pretrained models."""
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pkg_resources
from bioimageio.core import test_model
from bioimageio.spec import ValidationSummary, save_bioimageio_package
from torch import __version__, load, save

from careamics.config import Configuration, save_configuration
from careamics.config.support import SupportedArchitecture
from careamics.lightning_module import CAREamicsKiln
from careamics.utils import check_path_exists, get_careamics_home

from .bioimage import create_model_description


def load_pretrained(path: Union[Path, str]) -> Tuple[CAREamicsKiln, Configuration]:
    """
    Load a pretrained model from a checkpoint or a BioImage Model Zoo model.

    Expected formats are .ckpt, .zip, .pth or .pt files.

    Parameters
    ----------
    path : Union[Path, str]
        Path to the pretrained model.

    Returns
    -------
    CAREamicsKiln
        CAREamics model loaded from the checkpoint.

    Raises
    ------
    ValueError
        If the model format is not supported.
    """
    path = check_path_exists(path)

    if path.suffix == ".ckpt":
        # load checkpoint
        checkpoint: dict = load(path)

        # attempt to load algorithm parameters
        try:
            cfg_dict = checkpoint["hyper_parameters"]
        except KeyError as e:
            raise ValueError(
                f"Invalid checkpoint file. No `hyper_parameters` found in the "
                f"checkpoint: {checkpoint.keys()}"
            ) from e

        model = _load_from_checkpoint(path)

        return model, Configuration(**cfg_dict)

    elif path.suffix == ".zip":
        return _load_from_bmz(path)
    else:
        raise ValueError(
            f"Invalid model format. Expected .ckpt or .zip, " f"got {path.suffix}."
        )


def _load_from_checkpoint(path: Union[Path, str]) -> CAREamicsKiln:
    """
    Load a model from a checkpoint.

    Parameters
    ----------
    path : Union[Path, str]
        Path to the checkpoint.

    Returns
    -------
    CAREamicsKiln
        CAREamics model loaded from the checkpoint.
    """
    return CAREamicsKiln.load_from_checkpoint(path)


def _load_from_torch_dict(
    path: Union[Path, str]
) -> Tuple[CAREamicsKiln, Configuration]:
    """
    Load a model from a PyTorch dictionary.

    Parameters
    ----------
    path : Union[Path, str]
        Path to the PyTorch dictionary.

    Returns
    -------
    Tuple[CAREamicsKiln, Configuration]
        CAREamics model and Configuration loaded from the BioImage Model Zoo.
    """
    raise NotImplementedError(
        "Loading a model from a PyTorch dictionary is not implemented yet."
    )


def _load_from_bmz(
    path: Union[Path, str],
) -> Tuple[CAREamicsKiln, Configuration]:
    """
    Load a model from BioImage Model Zoo.

    Parameters
    ----------
    path : Union[Path, str]
        Path to the BioImage Model Zoo model.

    Returns
    -------
    Tuple[CAREamicsKiln, Configuration]
        CAREamics model and Configuration loaded from the BioImage Model Zoo.

    Raises
    ------
    NotImplementedError
        If the method is not implemented yet.
    """
    raise NotImplementedError(
        "Loading a model from BioImage Model Zoo is not implemented yet."
    )

    # load BMZ archive
    # extract model and call _load_from_torch_dict


def _export_state_dict(model: CAREamicsKiln, path: Union[Path, str]) -> Path:
    """
    Export the model state dictionary to a file.

    Parameters
    ----------
    model : CAREamicsKiln
        CAREamics model to export.
    path : Union[Path, str]
        Path to the file where to save the model state dictionary.

    Returns
    -------
    Path
        Path to the saved model state dictionary.
    """
    path = Path(path)

    # make sure it has the correct suffix
    if path.suffix not in ".pth":
        path = path.with_suffix(".pth")

    # save model state dictionary
    save(model.model.state_dict(), path)

    return path


def export_bmz(
    model: CAREamicsKiln,
    config: Configuration,
    path: Union[Path, str],
    name: str,
    general_description: str,
    authors: List[dict],
    inputs: Union[Path, str],
    outputs: Union[Path, str],
    data_description: Optional[str] = None,
    custom_description: Optional[str] = None,
) -> None:
    """
    Export the model to BioImage Model Zoo format.

    Parameters
    ----------
    model : CAREamicsKiln
        CAREamics model to export.
    config : Configuration
        Model configuration.
    path : Union[Path, str]
        Path to the output file.
    name : str
        Model name.
    general_description : str
        General description of the model.
    authors : List[dict]
        Authors of the model.
    inputs : Union[Path, str]
        Path to input .npy file.
    outputs : Union[Path, str]
        Path to output .npy file.
    data_description : Optional[str], optional
        Description of the data, by default None
    custom_description : Optional[str], optional
        Description of the custom algorithm, by default None
    """
    path = Path(path)

    # method is not compatible with Custom models
    if config.algorithm_config.model.architecture == SupportedArchitecture.CUSTOM:
        raise ValueError(
            "Exporting Custom models to BioImage Model Zoo format is not supported."
        )

    # make sure it has the correct suffix
    if path.suffix not in ".zip":
        path = path.with_suffix(".zip")

    # versions
    pytorch_version = __version__
    careamics_version = pkg_resources.get_distribution("careamics").version

    # create environment file
    env_path = get_careamics_home() / "environment.yml"
    env_path.write_text(
        f"name: careamics\n"
        f"dependencies:\n"
        f"  - python=3.8\n"
        f"  - pytorch={pytorch_version}\n"
        f"  - torchvision={pytorch_version}\n"
        f"  - pip\n"
        f"  - pip:\n"
        f"    - git+https://github.com/CAREamics/careamics.git@dl4mia\n"
    )
    # TODO from pip with package version

    # export configuration
    config_path = save_configuration(config, get_careamics_home())

    # export model state dictionary
    weight_path = _export_state_dict(model, get_careamics_home() / "weights.pth")

    # create model description
    model_description = create_model_description(
        config=config,
        name=name,
        general_description=general_description,
        authors=authors,
        inputs=inputs,
        outputs=outputs,
        weights_path=weight_path,
        torch_version=pytorch_version,
        careamics_version=careamics_version,
        config_path=config_path,
        env_path=env_path,
        data_description=data_description,
        custom_description=custom_description,
    )

    # test model description
    summary: ValidationSummary = test_model(model_description)
    if summary.status == "failed":
        raise ValueError(f"Model description test failed: {summary}")

    # save bmz model
    save_bioimageio_package(model_description, output_path=path)
