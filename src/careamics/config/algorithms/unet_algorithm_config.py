"""UNet-based algorithm Pydantic model."""

from pprint import pformat

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


    Attributes
    ----------
    algorithm : str
        Algorithm to use.
    loss : str
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
    algorithm: str
    """Algorithm name, as defined in SupportedAlgorithm."""

    loss: str
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

    def get_num_input_channels(self) -> int:
        """Get the number of input channels.

        Returns
        -------
        int
            Number of input channels.
        """
        return self.model.get_num_input_channels()

    def uses_batch_norm(self) -> bool:
        """Return whether the model uses batch normalization.

        Returns
        -------
        bool
            Whether the model uses batch normalization.
        """
        return self.model.uses_batch_norm()
