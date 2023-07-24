from pathlib import Path
from typing import Union

import yaml
from bioimageio.core import load_resource_description
from bioimageio.core.build_spec import build_model
from bioimageio.spec.model.raw_nodes import Model

from careamics_restoration.config.config import (
    Configuration,
    load_configuration,
    save_configuration,
)

PYTORCH_STATE_DICT = "pytorch_state_dict"
CAREAMICS_CONFIG = "careamics_config.yml"


def _load_model_rdf(name: str) -> dict:
    """Load an rdf(yaml) file given a model name."""
    name = name.lower()
    rdf_file = Path(__file__).parent.joinpath("rdfs").joinpath(name + ".yaml")
    if not rdf_file.exists:
        # a warning maybe?
        return {}

    with open(rdf_file) as f:
        rdf = yaml.safe_load(f)

    # add covers if available
    cover_folder = Path(__file__).parent.joinpath("covers")
    cover_images = list(cover_folder.glob(f"{name}_cover*.*"))
    if len(cover_images) > 0:
        rdf["covers"] = [str(img.absolute()) for img in cover_images]

    # add documentation (md file)
    doc = Path(__file__).parent.joinpath("docs").joinpath(f"{name}.md")
    if doc.exists() > 0:
        rdf["documentation"] = str(doc.absolute())

    return rdf


def get_default_model_specs(name: str) -> dict:
    """Return the default specs for given model's name."""
    return _load_model_rdf(name)


def build_zip_model(
    config: Configuration,
    model_specs: dict,
) -> Model:
    """Build bioimage model zip file from model specification data.

    Parameters
    ----------
        config (Configuration): Careamics' configuration,
        model_specs (dict): Model's specs to export.

    Return
    ------
        A bioimage raw Model
    """
    # save config file as attachments
    workdir = config.working_directory
    workdir.mkdir(parents=True, exist_ok=True)
    config_file = save_configuration(config, workdir.joinpath(CAREAMICS_CONFIG))
    # build model zip
    raw_model = build_model(
        root=str(Path(model_specs["output_path"]).parent.absolute()),
        attachments={"files": [str(config_file)]},
        **model_specs,
    )

    # delete config_file
    config_file.unlink()

    return raw_model


def import_bioimage_model(model_path: Union[str, Path]) -> Configuration:
    """Load configs and weights from a bioimage zip model.

    Parameters
    ----------
        model_path (Union[str, Path]): Path to the bioimage model

    Return
    ------
        Configuration instance
    """
    if isinstance(model_path, str):
        model_path = Path(model_path)
    # check the model extension (should be a zip file).
    if model_path.suffix != ".zip":
        raise ValueError("Invalid model format. Expected bioimage model zip file.")
    # load the model
    rdf = load_resource_description(model_path)
    config_file = None
    for file in rdf.attachments.files:
        if file.name == CAREAMICS_CONFIG:
            config_file = file
            break
    if config_file is not None:
        config = load_configuration(config_file)
        return config

    raise FileNotFoundError(f"No careamics config file was found in {model_path}.")
