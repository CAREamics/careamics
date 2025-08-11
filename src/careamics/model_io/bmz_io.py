"""Function to export to the BioImage Model Zoo format."""

import tempfile
from pathlib import Path
from typing import Union

import numpy as np
from bioimageio.core import load_model_description, test_model
from bioimageio.spec import ValidationSummary, save_bioimageio_package
from pydantic import HttpUrl
from torch import __version__ as PYTORCH_VERSION
from torch import load, save
from torchvision import __version__ as TORCHVISION_VERSION

from careamics.config import Configuration, load_configuration, save_configuration
from careamics.config.support import SupportedArchitecture
from careamics.lightning.lightning_module import FCNModule, VAEModule
from careamics.utils.version import get_careamics_version

from .bioimage import (
    create_env_text,
    create_model_description,
    extract_model_path,
)
from .bioimage.cover_factory import create_cover


def _export_state_dict(
    model: Union[FCNModule, VAEModule], path: Union[Path, str]
) -> Path:
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
    # we save through the torch model itself to avoid the initial "model." in the
    # layers naming, which is incompatible with the way the BMZ load torch state dicts
    save(model.model.state_dict(), path)

    return path


def _load_state_dict(
    model: Union[FCNModule, VAEModule], path: Union[Path, str]
) -> None:
    """
    Load a model from a state dictionary.

    Parameters
    ----------
    model : CAREamicsKiln
        CAREamics model to be updated with the weights.
    path : Union[Path, str]
        Path to the model state dictionary.
    """
    path = Path(path)

    # load model state dictionary
    # same as in _export_state_dict, we load through the torch model to be compatible
    # witht bioimageio.core expectations for a torch state dict
    state_dict = load(path)
    model.model.load_state_dict(state_dict)


# TODO break down in subfunctions
def export_to_bmz(
    model: Union[FCNModule, VAEModule],
    config: Configuration,
    path_to_archive: Union[Path, str],
    model_name: str,
    general_description: str,
    data_description: str,
    authors: list[dict],
    input_array: np.ndarray,
    output_array: np.ndarray,
    covers: list[Union[Path, str]] | None = None,
    channel_names: list[str] | None = None,
    model_version: str = "0.1.0",
) -> None:
    """Export the model to BioImage Model Zoo format.

    Arrays are expected to be SC(Z)YX with singleton dimensions allowed for S and C.

    `model_name` should consist of letters, numbers, dashes, underscores and parentheses
    only.

    Parameters
    ----------
    model : CAREamicsModule
        CAREamics model to export.
    config : Configuration
        Model configuration.
    path_to_archive : Union[Path, str]
        Path to the output file.
    model_name : str
        Model name.
    general_description : str
        General description of the model.
    data_description : str
        Description of the data the model was trained on.
    authors : list[dict]
        Authors of the model.
    input_array : np.ndarray
        Input array, should not have been normalized.
    output_array : np.ndarray
        Output array, should have been denormalized.
    covers : list of pathlib.Path or str, default=None
        Paths to the cover images.
    channel_names : Optional[list[str]], optional
        Channel names, by default None.
    model_version : str, default="0.1.0"
        Model version.
    """
    path_to_archive = Path(path_to_archive)

    if path_to_archive.suffix != ".zip":
        raise ValueError(
            f"Path to archive must point to a zip file, got {path_to_archive}."
        )

    if not path_to_archive.parent.exists():
        path_to_archive.parent.mkdir(parents=True, exist_ok=True)

    # versions
    careamics_version = get_careamics_version()

    # save files in temporary folder
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_path = Path(tmpdirname)

        # create environment file
        # TODO move in bioimage module
        env_path = temp_path / "environment.yml"
        env_path.write_text(create_env_text(PYTORCH_VERSION, TORCHVISION_VERSION))

        # export input and ouputs
        inputs = temp_path / "inputs.npy"
        np.save(inputs, input_array)
        outputs = temp_path / "outputs.npy"
        np.save(outputs, output_array)

        # export configuration
        config_path = save_configuration(config, temp_path / "careamics.yaml")

        # export model state dictionary
        weight_path = _export_state_dict(model, temp_path / "weights.pth")

        # export cover if necesary
        if covers is None:
            covers = [create_cover(temp_path, input_array, output_array)]

        # create model description
        model_description = create_model_description(
            config=config,
            name=model_name,
            general_description=general_description,
            data_description=data_description,
            authors=authors,
            inputs=inputs,
            outputs=outputs,
            weights_path=weight_path,
            torch_version=PYTORCH_VERSION,
            careamics_version=careamics_version,
            config_path=config_path,
            env_path=env_path,
            covers=covers,
            channel_names=channel_names,
            model_version=model_version,
        )

        # test model description
        test_kwargs = {}
        if hasattr(model_description, "config") and isinstance(
            model_description.config, dict
        ):
            bioimageio_config = model_description.config.get("bioimageio", {})
            test_kwargs = bioimageio_config.get("test_kwargs", {}).get(
                "pytorch_state_dict", {}
            )

        summary: ValidationSummary = test_model(model_description, **test_kwargs)
        if summary.status == "failed":
            raise ValueError(f"Model description test failed: {summary}")

        # save bmz model
        save_bioimageio_package(model_description, output_path=path_to_archive)


def load_from_bmz(
    path: Union[Path, str, HttpUrl],
) -> tuple[Union[FCNModule, VAEModule], Configuration]:
    """Load a model from a BioImage Model Zoo archive.

    Parameters
    ----------
    path : Path, str or HttpUrl
        Path to the BioImage Model Zoo archive. A Http URL must point to a downloadable
        location.

    Returns
    -------
    FCNModel or VAEModel
        The loaded CAREamics model.
    Configuration
        The loaded CAREamics configuration.

    Raises
    ------
    ValueError
        If the path is not a zip file.
    """
    # load description, this creates an unzipped folder next to the archive
    model_desc = load_model_description(path)

    # extract paths
    weights_path, config_path = extract_model_path(model_desc)

    # load configuration
    config = load_configuration(config_path)

    # create careamics lightning module
    if config.algorithm_config.model.architecture == SupportedArchitecture.UNET:
        model = FCNModule(algorithm_config=config.algorithm_config)
    elif config.algorithm_config.model.architecture == SupportedArchitecture.LVAE:
        model = VAEModule(algorithm_config=config.algorithm_config)
    else:
        raise ValueError(
            f"Unsupported architecture {config.algorithm_config.model.architecture}"
        )  # TODO ugly ?

    # load model state dictionary
    _load_state_dict(model, weights_path)

    return model, config
