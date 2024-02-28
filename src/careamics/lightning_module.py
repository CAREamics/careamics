from typing import Any, Union

import pytorch_lightning as L
from torch import nn

from careamics.config import AlgorithmModel
from careamics.config.support import (
    SupportedAlgorithm,
    SupportedArchitecture,
    SupportedLoss,
    SupportedOptimizer,
    SupportedScheduler,
)
from careamics.losses import loss_factory
from careamics.models.model_factory import model_factory
from careamics.utils.torch_utils import get_optimizer, get_scheduler


class CAREamicsKiln(L.LightningModule):
    """CAREamics internal Lightning module class.

    This class is configured using an AlgorithmModel instance, parameterizing the deep
    learning model, and defining training and validation steps.
    """

    def __init__(self, algorithm_config: AlgorithmModel) -> None:
        super().__init__()

        # create model and loss function
        self.model: nn.Module = model_factory(algorithm_config.model)
        self.loss_func = loss_factory(algorithm_config.loss)

        # save optimizer and lr_scheduler names and parameters
        self.optimizer_name = algorithm_config.optimizer.name
        self.optimizer_params = algorithm_config.optimizer.parameters
        self.lr_scheduler_name = algorithm_config.lr_scheduler.name
        self.lr_scheduler_params = algorithm_config.lr_scheduler.parameters

    def forward(self, x: Any) -> Any:
        return self.model(x)

    def training_step(self, batch, batch_idx) -> Any:
        x, *aux = batch
        out = self.model(x)
        loss = self.loss_func(out, *aux)
        return loss

    def validation_step(self, batch, batch_idx):
        x, *aux = batch
        out = self.model(x)
        val_loss = self.loss_func(out, *aux)

        # log validation loss
        self.log("val_loss", val_loss)

    def predict_step(self, batch, batch_idx) -> Any:
        x, *aux = batch
        out = self.model(x)
        return out, aux

    def configure_optimizers(self) -> Any:
        # instantiate optimizer
        optimizer_func = get_optimizer(self.optimizer_name)
        optimizer = optimizer_func(self.model.parameters(), **self.optimizer_params)

        # and scheduler
        scheduler_func = get_scheduler(self.lr_scheduler_name)
        scheduler = scheduler_func(optimizer, **self.lr_scheduler_params)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",  # otherwise triggers MisconfigurationException
        }


class CAREamicsModule(CAREamicsKiln):
    """Class defining the API for CAREamics Lightning layer.

    This class exposes parameters used to create an AlgorithmModel instance, triggering
    parameters validation.

    Parameters
    ----------
    algorithm : Union[SupportedAlgorithm, str]
        Algorithm to use for training (see SupportedAlgorithm).
    loss : Union[SupportedLoss, str]
        Loss function to use for training (see SupportedLoss).
    architecture : Union[SupportedArchitecture, str]
        Model architecture to use for training (see SupportedArchitecture).
    model_parameters : dict, optional
        Model parameters to use for training, by default {}. Model parameters are
        defined in the relevant `torch.nn.Module` class, or Pyddantic model (see
        `careamics.config.architectures`).
    optimizer : Union[SupportedOptimizer, str], optional
        Optimizer to use for training, by default "Adam" (see SupportedOptimizer).
    optimizer_parameters : dict, optional
        Optimizer parameters to use for training, as defined in `torch.optim`, by
        default {}.
    lr_scheduler : Union[SupportedScheduler, str], optional
        Learning rate scheduler to use for training, by default "ReduceLROnPlateau"
        (see SupportedScheduler).
    lr_scheduler_parameters : dict, optional
        Learning rate scheduler parameters to use for training, as defined in
        `torch.optim`, by default {}.
    """

    def __init__(
        self,
        algorithm: Union[SupportedAlgorithm, str],
        loss: Union[SupportedLoss, str],
        architecture: Union[SupportedArchitecture, str],
        model_parameters: dict = {},
        optimizer: Union[SupportedOptimizer, str] = "Adam",
        optimizer_parameters: dict = {},
        lr_scheduler: Union[SupportedScheduler, str] = "ReduceLROnPlateau",
        lr_scheduler_parameters: dict = {},
    ) -> None:

        # create a AlgorithmModel compatible dictionary
        algorithm_configuration = {
            "algorithm": algorithm,
            "loss": loss,
            "optimizer": {
                "name": optimizer,
                "parameters": optimizer_parameters,
            },
            "lr_scheduler": {
                "name": lr_scheduler,
                "parameters": lr_scheduler_parameters,
            },
        }
        model_configuration = {"architecture": architecture}
        model_configuration.update(model_parameters)

        # add model parameters to algorithm configuration
        algorithm_configuration["model"] = model_configuration

        # call the parent init using an AlgorithmModel instance
        super().__init__(AlgorithmModel(**algorithm_configuration))
