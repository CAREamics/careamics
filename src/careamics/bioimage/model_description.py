from pathlib import Path
from typing import List, Optional, Union, Tuple

from bioimageio.spec.model.v0_5 import (
    Author,
    AxisId,
    AxisBase,
    BatchAxis,
    ChannelAxis,
    FileDescr,
    Identifier,
    InputTensorDescr,
    ModelDescr,
    OutputTensorDescr,
    ParameterizedSize,
    SpaceInputAxis,
    SpaceOutputAxis,
    TensorId,
    DocumentationSource,
)

from .readme_factory import readme_factory
from careamics import CAREamist
from careamics.config import DataModel, save_configuration
from careamics.utils import cwd, get_careamics_home



def _create_axes(
        data_config: DataModel, 
        is_input: bool = True,
        channel_names: Optional[List[str]] = None,
) -> List[AxisBase]:
    """Create axes description.

    Parameters
    ----------
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
                ChannelAxis(channel_names=[Identifier(name) for name in channel_names]))
        else:
            raise ValueError(
                f"Channel names must be provided if channel axis is present, axes: "
                f"{data_config.axes}."
            )
    else:
        axes_model.append(ChannelAxis(channel_names=[Identifier("raw")]))

    # spatial axes
    for axes in data_config.axes:
        if axes in ['X', 'Y', 'Z']:

            if is_input:
                axes_model.append(
                    SpaceInputAxis(
                        id=AxisId(axes.lower()), 
                        size=ParameterizedSize(min=16, step=8) # TODO check the min/step
                    )
                )
            else:
                axes_model.append(
                    SpaceOutputAxis(
                        id=AxisId(axes.lower()), 
                        size=ParameterizedSize(min=16, step=8)
                    )
                )

    return axes_model

def _create_inputs_ouputs(
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
    input_axes = _create_axes(data_config)
    output_axes = _create_axes(data_config)
    input_descr = InputTensorDescr(
        id=TensorId("raw"), 
        axes=input_axes, 
        test_tensor=FileDescr(source=input_path)
    )
    output_descr = OutputTensorDescr(
        id=TensorId("pred"), 
        axes=output_axes, 
        test_tensor=FileDescr(source=output_path)
        )
    
    return input_descr, output_descr


def create_model_description(
        name: str,
        general_description: str,
        careamist: CAREamist,
        authors: List[Author],
        inputs: Union[Path, str],
        outputs: Union[Path, str],
        weights: Union[Path, str],
        data_description: Optional[str] = None,
        custom_description: Optional[str] = None
) -> ModelDescr:
    doc = readme_factory(
        careamist.cfg,
        data_description=data_description,
        custom_description=custom_description
    )

    input_descr, output_descr = _create_inputs_ouputs(
        careamist.cfg.data,
        input_path=inputs,
        output_path=outputs,
    )

    # export configuration
    with cwd(get_careamics_home()):
        config_path = save_configuration(careamist.cfg, get_careamics_home())


    model = ModelDescr(
        name=name,
        authors=authors,
        description=general_description,
        documentation=doc,
        inputs=[input_descr],
        outputs=[output_descr],
        tags=careamist.cfg.get_algorithm_keywords(),
        links=[
            "https://github.com/CAREamics/careamics",
            "https://careamics.github.io/latest/"
        ],
        license="BSD-3-Clause",
        version= '0.1.0',
        weights=weights,
        attachments=[config_path]
    )

    return model