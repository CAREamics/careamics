"""Algorithm configuration."""

from __future__ import annotations

from pprint import pformat
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from .architectures import CustomModel, LVAEModel
from .likelihood_model import GaussianLikelihoodModel, NMLikelihoodModel
from .nm_model import GMNMModel
from .optimizer_models import LrSchedulerModel, OptimizerModel


class VAEAlgorithmConfig(BaseModel):
    """Algorithm configuration.

    This Pydantic model validates the parameters governing the components of the
    training algorithm: which algorithm, loss function, model architecture, optimizer,
    and learning rate scheduler to use.

    Currently, we only support N2V, CARE, N2N and custom models. The `n2v` algorithm is
    only compatible with `n2v` loss and `UNet` architecture. The `custom` algorithm
    allows you to register your own architecture and select it using its name as
    `name` in the custom pydantic model.

    Attributes
    ----------
    algorithm : Literal["n2v", "custom"]
        Algorithm to use.
    loss : Literal["n2v", "mae", "mse"]
        Loss function to use.
    model : Union[UNetModel, LVAEModel, CustomModel]
        Model architecture to use.
    optimizer : OptimizerModel, optional
        Optimizer to use.
    lr_scheduler : LrSchedulerModel, optional
        Learning rate scheduler to use.

    Raises
    ------
    ValueError
        Algorithm parameter type validation errors.
    ValueError
        If the algorithm, loss and model are not compatible.

    Examples
    --------
    Minimum example:
    >>> from careamics.config import AlgorithmConfig
    >>> config_dict = {
    ...     "algorithm": "n2v",
    ...     "loss": "n2v",
    ...     "model": {
    ...         "architecture": "UNet",
    ...     }
    ... }
    >>> config = AlgorithmConfig(**config_dict)

    Using a custom model:
    >>> from torch import nn, ones
    >>> from careamics.config import AlgorithmConfig, register_model
    ...
    >>> @register_model(name="linear_model")
    ... class LinearModel(nn.Module):
    ...    def __init__(self, in_features, out_features, *args, **kwargs):
    ...        super().__init__()
    ...        self.in_features = in_features
    ...        self.out_features = out_features
    ...        self.weight = nn.Parameter(ones(in_features, out_features))
    ...        self.bias = nn.Parameter(ones(out_features))
    ...    def forward(self, input):
    ...        return (input @ self.weight) + self.bias
    ...
    >>> config_dict = {
    ...     "algorithm": "custom",
    ...     "loss": "mse",
    ...     "model": {
    ...         "architecture": "Custom",
    ...         "name": "linear_model",
    ...         "in_features": 10,
    ...         "out_features": 5,
    ...     }
    ... }
    >>> config = AlgorithmConfig(**config_dict)
    """

    # Pydantic class configuration
    model_config = ConfigDict(
        protected_namespaces=(),  # allows to use model_* as a field name
        validate_assignment=True,
        extra="allow",
    )

    # Mandatory fields
    # defined in SupportedAlgorithm
    algorithm_type: Literal["vae"]
    algorithm: Literal["musplit", "denoisplit", "custom"]
    loss: Literal["musplit_loss", "denoisplit_loss"]
    model: Union[LVAEModel, CustomModel] = Field(discriminator="architecture")

    noise_model: GMNMModel = Field(discriminator="model_type")
    likelihood_model: Union[GaussianLikelihoodModel, NMLikelihoodModel] = Field(
        discriminator="model_type"
    )

    # Optional fields
    optimizer: OptimizerModel = OptimizerModel()
    """Optimizer to use, defined in SupportedOptimizer."""

    lr_scheduler: LrSchedulerModel = LrSchedulerModel()

    @model_validator(mode="after")
    def algorithm_cross_validation(self: Self) -> Self:
        """Validate the algorithm model based on `algorithm`.

        Returns
        -------
        Self
            The validated model.
        """
        # musplit
        if self.algorithm == "musplit":
            if self.loss != "musplit_loss":
                raise ValueError(
                    f"Algorithm {self.algorithm} only supports loss `musplit`."
                )
        # TODO add more checks

        if self.algorithm == "denoisplit":
            if self.loss == "denoisplit_loss":
                raise ValueError(
                    f"Algorithm {self.algorithm} only supports loss `denoisplit_loss`."
                )

        return self

    def __str__(self) -> str:
        """Pretty string representing the configuration.

        Returns
        -------
        str
            Pretty string.
        """
        return pformat(self.model_dump())
