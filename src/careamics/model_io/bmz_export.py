"""Function to export to the BioImage Model Zoo format."""
from pathlib import Path
from typing import List, Optional, Union
import tempfile

import numpy as np
import pkg_resources
from bioimageio.core import test_model
from bioimageio.spec import ValidationSummary, save_bioimageio_package
from torch import __version__

from careamics.config import Configuration, save_configuration
from careamics.config.support import SupportedArchitecture
from careamics.lightning_module import CAREamicsKiln

from .bioimage import create_model_description
from .model_io_utils import export_state_dict


# TODO break down in subfunctions
def export_to_bmz(
    model: CAREamicsKiln,
    config: Configuration,
    path: Union[Path, str],
    name: str,
    general_description: str,
    authors: List[dict],
    input_array: np.ndarray,
    output_array: np.ndarray,
    channel_names: Optional[List[str]] = None,
    data_description: Optional[str] = None,
) -> None:
    """Export the model to BioImage Model Zoo format.

    Arrays are expected to be SC(Z)YX with singleton dimensions allowed for S and C.

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
    input_array : np.ndarray
        Input array.
    output_array : np.ndarray
        Output array.
    channel_names : Optional[List[str]], optional
        Channel names, by default None
    data_description : Optional[str], optional
        Description of the data, by default None
        
    Raises
    ------
    ValueError
        If the model is a Custom model.
    """
    path = Path(path)

    # method is not compatible with Custom models
    if config.algorithm_config.model.architecture == SupportedArchitecture.CUSTOM:
        raise ValueError(
            "Exporting Custom models to BioImage Model Zoo format is not supported."
        )

    # make sure that input and output arrays have the same shape
    assert input_array.shape == output_array.shape, \
        f"Input ({input_array.shape}) and output ({output_array.shape}) arrays " \
        f"have different shapes"

    # make sure it has the correct suffix
    if path.suffix not in ".zip":
        path = path.with_suffix(".zip")

    # versions
    pytorch_version = __version__
    careamics_version = pkg_resources.get_distribution("careamics").version

    # save files in temporary folder
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_path = Path(tmpdirname)

        # create environment file
        # TODO move in bioimage module
        env_path = temp_path / "environment.yml"
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

        # export input and ouputs
        inputs = temp_path / "inputs.npy"
        np.save(inputs, input_array)
        outputs = temp_path / "outputs.npy"
        np.save(outputs, output_array)

        # export configuration
        config_path = save_configuration(config, temp_path)

        # export model state dictionary
        weight_path = export_state_dict(model, temp_path / "weights.pth")

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
            channel_names=channel_names,
            data_description=data_description,
        )

        # test model description
        summary: ValidationSummary = test_model(model_description)
        if summary.status == "failed":
            raise ValueError(f"Model description test failed: {summary}")

        # save bmz model
        save_bioimageio_package(model_description, output_path=path)
