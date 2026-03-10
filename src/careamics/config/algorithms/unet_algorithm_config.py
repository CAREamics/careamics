"""UNet-based algorithm Pydantic model."""

from pprint import pformat
from typing import Literal

from pydantic import BaseModel, ConfigDict

from careamics.config.architectures import UNetConfig
from careamics.config.lightning.optimizer_configs import (
    LrSchedulerConfig,
    OptimizerConfig,
)


class UNetBasedAlgorithm(BaseModel):
    """General UNet-based algorithm configuration.

    This Pydantic model validates the parameters governing the components of the
    training algorithm: which algorithm, loss function, model architecture, optimizer,
    and learning rate scheduler to use.

    Currently, we only support N2V, CARE, N2N, and PN2V algorithms. In order to train
    these algorithms, use the corresponding configuration child classes (e.g.
    `N2VAlgorithm`) to ensure coherent parameters (e.g. specific losses).


    Attributes
    ----------
    algorithm : {"n2v", "care", "n2n", "pn2v"}
        Algorithm to use.
    loss : {"n2v", "mae", "mse"}
        Loss function to use.
    model : UNetConfig
        Model architecture to use.
    optimizer : OptimizerConfig, optional
        Optimizer to use.
    lr_scheduler : LrSchedulerConfig, optional
        Learning rate scheduler to use.

    Raises
    ------
    ValueError
        Algorithm parameter type validation errors.
    ValueError
        If the algorithm, loss and model are not compatible.
    """

    # Pydantic class configuration
    model_config = ConfigDict(
        protected_namespaces=(),  # allows to use model_* as a field name
        validate_assignment=True,
        extra="allow",
    )

    # Mandatory fields
    algorithm: Literal["n2v", "care", "n2n", "pn2v"]
    """Algorithm name, as defined in SupportedAlgorithm."""

    loss: Literal["n2v", "mae", "mse", "pn2v"]
    """Loss function to use, as defined in SupportedLoss."""

    model: UNetConfig
    """UNet model configuration."""

    # Optional fields
    optimizer: OptimizerConfig = OptimizerConfig()
    """Optimizer to use, defined in SupportedOptimizer."""

    lr_scheduler: LrSchedulerConfig = LrSchedulerConfig()
    """Learning rate scheduler to use, defined in SupportedLrScheduler."""

    def __str__(self) -> str:
        """Pretty string representing the configuration.

        Returns
        -------
        str
            Pretty string.
        """
        return pformat(self.model_dump())

    @classmethod
    def get_compatible_algorithms(cls) -> list[str]:
        """Get the list of compatible algorithms.

        Returns
        -------
        list of str
            List of compatible algorithms.
        """
        return ["n2v", "care", "n2n", "pn2v"]
