from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from bioimageio.spec.model.v0_5 import (
    ArchitectureFromLibraryDescr,
    Author,
    AxisBase,
    AxisId,
    BatchAxis,
    ChannelAxis,
    EnvironmentFileDescr,
    FileDescr,
    FixedZeroMeanUnitVarianceDescr,
    FixedZeroMeanUnitVarianceKwargs,
    Identifier,
    InputTensorDescr,
    ModelDescr,
    OutputTensorDescr,
    PytorchStateDictWeightsDescr,
    SpaceInputAxis,
    SpaceOutputAxis,
    TensorId,
    Version,
    WeightsDescr,
)

from careamics.config import Configuration, DataModel

from .readme_factory import readme_factory


def _create_axes(
    array: np.ndarray,
    data_config: DataModel,
    is_input: bool = True,
    channel_names: Optional[List[str]] = None,
) -> List[AxisBase]:
    """Create axes description.

    Array shape is expected to be SC(Z)YX.

    Parameters
    ----------
    array : np.ndarray
        Array.
    config : DataModel
        CAREamics data configuration
    is_input : bool, optional
        Whether the axes are input axes, by default True
    channel_names : Optional[List[str]], optional
        Channel names, by default None

    Returns
    -------
    List[AxisBase]
        List of axes description

    Raises
    ------
    ValueError
        If channel names are not provided when channel axis is present
    """
    # batch is always present
    axes_model = [BatchAxis()]

    if "C" in data_config.axes:
        if channel_names is not None:
            axes_model.append(
                ChannelAxis(channel_names=[Identifier(name) for name in channel_names])
            )
        else:
            raise ValueError(
                f"Channel names must be provided if channel axis is present, axes: "
                f"{data_config.axes}."
            )
    else:
        # singleton channel
        axes_model.append(ChannelAxis(channel_names=[Identifier("channel")]))

    # spatial axes
    for ind, axes in enumerate(data_config.axes):
        if axes in ["X", "Y", "Z"]:
            if is_input:
                axes_model.append(
                    SpaceInputAxis(id=AxisId(axes.lower()), size=array.shape[2 + ind])
                )
            else:
                axes_model.append(
                    SpaceOutputAxis(id=AxisId(axes.lower()), size=array.shape[2 + ind])
                )

    return axes_model


def _create_inputs_ouputs(
    input_array: np.ndarray,
    output_array: np.ndarray,
    data_config: DataModel,
    input_path: Union[Path, str],
    output_path: Union[Path, str],
) -> Tuple[InputTensorDescr, OutputTensorDescr]:
    """Create input and output tensor description.

    Input and output paths must point to a `.npy` file.

    Parameters
    ----------
    data_config : DataModel
        CAREamics data configuration
    input_path : Union[Path, str]
        Path to input .npy file
    output_path : Union[Path, str]
        Path to output .npy file

    Returns
    -------
    Tuple[InputTensorDescr, OutputTensorDescr]
        Input and output tensor descriptions
    """
    input_axes = _create_axes(input_array, data_config)
    output_axes = _create_axes(output_array, data_config, is_input=False)

    # mean and std
    mean = data_config.mean
    std = data_config.std

    # and the mean and std required to invert the normalization
    inv_mean = -mean / std
    inv_std = 1 / std

    # create input/output descriptions
    input_descr = InputTensorDescr(
        id=TensorId("input"),
        axes=input_axes,
        test_tensor=FileDescr(source=input_path),
        preprocessing=[
            FixedZeroMeanUnitVarianceDescr(
                kwargs=FixedZeroMeanUnitVarianceKwargs(mean=mean, std=std)
            )
        ],
    )
    output_descr = OutputTensorDescr(
        id=TensorId("prediction"),
        axes=output_axes,
        test_tensor=FileDescr(source=output_path),
        postprocessing=[
            FixedZeroMeanUnitVarianceDescr(
                kwargs=FixedZeroMeanUnitVarianceKwargs(  # invert normalization
                    mean=inv_mean, std=inv_std
                )
            )
        ],
    )

    return input_descr, output_descr


def create_model_description(
    config: Configuration,
    name: str,
    general_description: str,
    authors: List[Author],
    inputs: Union[Path, str],
    outputs: Union[Path, str],
    weights_path: Union[Path, str],
    torch_version: str,
    careamics_version: str,
    config_path: Union[Path, str],
    env_path: Union[Path, str],
    data_description: Optional[str] = None,
    custom_description: Optional[str] = None,
) -> ModelDescr:
    """Create model description.

    Parameters
    ----------
    careamist : CAREamist
        CAREamist instance.
    name : str
        Name fo the model.
    general_description : str
        General description of the model.
    authors : List[Author]
        Authors of the model.
    inputs : Union[Path, str]
        Path to input .npy file.
    outputs : Union[Path, str]
        Path to output .npy file.
    weights_path : Union[Path, str]
        Path to model weights.
    torch_version : str
        Pytorch version.
    config_path : Union[Path, str]
        Path to model configuration.
    env_path : Union[Path, str]
        Path to environment file.
    data_description : Optional[str], optional
        Description of the data, by default None
    custom_description : Optional[str], optional
        Description of the custom algorithm, by default None

    Returns
    -------
    ModelDescr
        Model description.
    """
    # documentation
    doc = readme_factory(
        config,
        careamics_version=careamics_version,
        data_description=data_description,
        custom_description=custom_description,
    )

    # inputs, outputs
    input_descr, output_descr = _create_inputs_ouputs(
        input_array=np.load(inputs),
        output_array=np.load(outputs),
        data_config=config.data_config,
        input_path=inputs,
        output_path=outputs,
    )

    # weights description
    architecture_descr = ArchitectureFromLibraryDescr(
        import_from="careamics.models",
        callable=f"{config.algorithm_config.model.architecture}",
        kwargs=config.algorithm_config.model.model_dump(),
    )

    weights_descr = WeightsDescr(
        pytorch_state_dict=PytorchStateDictWeightsDescr(
            source=weights_path,
            architecture=architecture_descr,
            pytorch_version=Version(torch_version),
            dependencies=EnvironmentFileDescr(source=env_path),
        ),
    )

    # overall model description
    model = ModelDescr(
        name=name,
        authors=authors,
        description=general_description,
        documentation=doc,
        inputs=[input_descr],
        outputs=[output_descr],
        tags=config.get_algorithm_keywords(),
        links=[
            "https://github.com/CAREamics/careamics",
            "https://careamics.github.io/latest/",
        ],
        license="BSD-3-Clause",
        version="0.1.0",
        weights=weights_descr,
        attachments=[FileDescr(source=config_path)],
        cite=config.get_algorithm_citations(),
    )

    return model
