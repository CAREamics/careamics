from pathlib import Path
from typing import Optional, Union

# from bioimageio.spec.model import raw_nodes as model
# from bioimageio.spec.rdf import raw_nodes as rdf
# from bioimageio.spec.shared.raw_nodes import RawNode
from bioimageio.spec.model.raw_nodes import Model
from bioimageio.core.build_spec import build_model

from careamics_restoration.config.config import Configuration
# from careamics_restoration.config.algorithm import Losses
from .rdfs import BIOIMAGE_MODEL_RDFS


FORMAT_VERSION = "0.4.9"
PYTORCH_STATE_DICT = "pytorch_state_dict"
TORCH_SCRIPT = "torchscript"


def build_zip_model(
    config: Configuration,
    weights: Union[str, Path],
    sample_inputs: Union[str, Path], sample_outputs: Union[str, Path],
    export_name: Optional[str] = None,
    model_rdf: Optional[dict] = None
) -> Model:
    """Build bioimage model zip file from model specification data.

    Parameters
    ----------
        config (Configuration): Careamics' configuration,
        weights (Union[str, Path]): Path to model's weights file
        sample_inputs (Union[str, Path]): Model's test inputs (numpy files)
        sample_outputs (Union[str, Path]): Model's test outputs (numpy files)
        model_rdf (Optional[dict], optional): Model's specs to export.
        export_name (Optional[str]): The export zip file name,
        If None then the default specs will be used. Defaults to None.

    Return
    ------
        A bioimage raw Model
    """
    if model_rdf is None:
        # load default model rdf
        model_rdf = BIOIMAGE_MODEL_RDFS[config.algorithm.loss]

    workdir = config.working_directory
    workdir.mkdir(parents=True, exist_ok=True)
    zip_file_path = workdir.joinpath(
        export_name or config.algorithm.loss.value + ".zip"
    )

    raw_model = build_model(
        output_path=zip_file_path,
        name=model_rdf["name"],
        weight_type=PYTORCH_STATE_DICT,
        weight_uri=str(weights),
        # architecture=model_source_code,
        # model_kwargs=model_source_sha256,
        test_inputs=[str(item) for item in sample_inputs],
        test_outputs=[str(item) for item in sample_outputs],
        input_names=[_input["name"] for _input in model_rdf["inputs"]],
        input_axes=[_input["axes"] for _input in model_rdf["inputs"]],
        preprocessing=[_input.get("preprocessing") for _input in model_rdf["inputs"]],
        output_names=[output["name"] for output in model_rdf["outputs"]],
        output_axes=[output["axes"] for output in model_rdf["outputs"]],
        halo=[output.get("halo") for output in model_rdf["outputs"]],
        postprocessing=[
            output.get("postprocessing") for output in model_rdf["outputs"]
        ],
        authors=model_rdf["authors"],
        cite=model_rdf["cite"],
        documentation=model_rdf["documentation"],
        description=model_rdf["description"],
        license=model_rdf["license"],
        # optionals
        covers=model_rdf.get("covers"),
        tags=model_rdf.get("tags"),
        root=Path(zip_file_path).parent,
    )

    return raw_model
