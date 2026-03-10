"""Convenience function to create algorithm configurations."""

from typing import Annotated, Any, Literal, Union

from pydantic import Field, TypeAdapter

from careamics.config.algorithms import (
    CAREAlgorithm,
    N2NAlgorithm,
    N2VAlgorithm,
    # PN2VAlgorithm,  # TODO not yet compatible with NG Dataset
)
from careamics.config.architectures import UNetConfig
from careamics.config.support.supported_architectures import SupportedArchitecture


# TODO rename so that it does not bear the same name as the module?
def algorithm_factory(
    algorithm: dict[str, Any],
) -> Union[N2VAlgorithm, N2NAlgorithm, CAREAlgorithm]:
    """
    Create an algorithm model for training CAREamics.

    Parameters
    ----------
    algorithm : dict
        Algorithm dictionary.

    Returns
    -------
    N2VAlgorithm or N2NAlgorithm or CAREAlgorithm
        Algorithm model for training CAREamics.
    """
    adapter: TypeAdapter = TypeAdapter(
        Annotated[
            Union[N2VAlgorithm, N2NAlgorithm, CAREAlgorithm],
            Field(discriminator="algorithm"),
        ]
    )
    return adapter.validate_python(algorithm)


def create_algorithm_configuration(
    dimensions: Literal[2, 3],
    algorithm: Literal["n2v", "care", "n2n"],
    loss: Literal["n2v", "mae", "mse"],
    independent_channels: bool,
    n_channels_in: int,
    n_channels_out: int,
    use_n2v2: bool = False,
    model_params: dict | None = None,
    optimizer: Literal["Adam", "Adamax", "SGD"] = "Adam",
    optimizer_params: dict[str, Any] | None = None,
    lr_scheduler: Literal["ReduceLROnPlateau", "StepLR"] = "ReduceLROnPlateau",
    lr_scheduler_params: dict[str, Any] | None = None,
) -> dict:
    """
    Create a dictionary with the parameters of the algorithm model.

    Parameters
    ----------
    dimensions : {2, 3}
        Dimension of the model, either 2D or 3D.
    algorithm : {"n2v", "care", "n2n"}
        Algorithm to use.
    loss : {"n2v", "mae", "mse"}
        Loss function to use.
    independent_channels : bool
        Whether to train all channels independently.
    n_channels_in : int
        Number of input channels.
    n_channels_out : int
        Number of output channels.
    use_n2v2 : bool, default=false
        Whether to use N2V2.
    model_params : dict, default=None
        UNetModel parameters.
    optimizer : {"Adam", "Adamax", "SGD"}, default="Adam"
        Optimizer to use.
    optimizer_params : dict, default=None
        Parameters for the optimizer, see PyTorch documentation for more details.
    lr_scheduler : {"ReduceLROnPlateau", "StepLR"}, default="ReduceLROnPlateau"
        Learning rate scheduler to use.
    lr_scheduler_params : dict, default=None
        Parameters for the learning rate scheduler, see PyTorch documentation for more
        details.


    Returns
    -------
    dict
        Algorithm model as dictionnary with the specified parameters.
    """
    # create dictionary to ensure priority of explicit parameters over model_params
    # and prevent multiple same parameters being passed to UNetConfig
    model_params = {} if model_params is None else model_params
    model_params["n2v2"] = use_n2v2
    model_params["conv_dims"] = dimensions
    model_params["in_channels"] = n_channels_in
    model_params["num_classes"] = n_channels_out
    model_params["independent_channels"] = independent_channels

    unet_model = UNetConfig(
        architecture=SupportedArchitecture.UNET.value,
        **model_params,
    )

    return {
        "algorithm": algorithm,
        "loss": loss,
        "model": unet_model,
        "optimizer": {
            "name": optimizer,
            "parameters": {} if optimizer_params is None else optimizer_params,
        },
        "lr_scheduler": {
            "name": lr_scheduler,
            "parameters": {} if lr_scheduler_params is None else lr_scheduler_params,
        },
    }
