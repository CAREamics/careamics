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


def _create_unet_configuration(
    axes: str,
    n_channels_in: int,
    n_channels_out: int,
    independent_channels: bool,
    use_n2v2: bool,
    model_params: dict[str, Any] | None = None,
) -> UNetConfig:
    """
    Create a dictionary with the parameters of the UNet model.

    Parameters
    ----------
    axes : str
        Axes of the data.
    n_channels_in : int
        Number of input channels.
    n_channels_out : int
        Number of output channels.
    independent_channels : bool
        Whether to train all channels independently.
    use_n2v2 : bool
        Whether to use N2V2.
    model_params : dict
        UNetModel parameters.

    Returns
    -------
    UNetModel
        UNet model with the specified parameters.
    """
    if model_params is None:
        model_params = {}

    model_params["n2v2"] = use_n2v2
    model_params["conv_dims"] = 3 if "Z" in axes else 2
    model_params["in_channels"] = n_channels_in
    model_params["num_classes"] = n_channels_out
    model_params["independent_channels"] = independent_channels

    return UNetConfig(
        architecture=SupportedArchitecture.UNET.value,
        **model_params,
    )


def create_algorithm_configuration(
    axes: str,
    algorithm: Literal["n2v", "care", "n2n", "pn2v"],
    loss: Literal["n2v", "mae", "mse", "pn2v"],
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
    axes : str
        Axes of the data.
    algorithm : {"n2v", "care", "n2n", "pn2v"}
        Algorithm to use.
    loss : {"n2v", "mae", "mse", "pn2v"}
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
    # model
    unet_model = _create_unet_configuration(
        axes=axes,
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
        independent_channels=independent_channels,
        use_n2v2=use_n2v2,
        model_params=model_params,
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
