"""CAREamics Lightning module."""

from typing import Any, Optional, Union

import pytorch_lightning as L
from torch import Tensor, nn

from careamics.config import AlgorithmConfig
from careamics.config.support import (
    SupportedAlgorithm,
    SupportedArchitecture,
    SupportedLoss,
    SupportedOptimizer,
    SupportedScheduler,
)
from careamics.losses import loss_factory
from careamics.models.model_factory import model_factory
from careamics.transforms import Denormalize, ImageRestorationTTA
from careamics.utils.torch_utils import get_optimizer, get_scheduler


class CAREamicsModule(L.LightningModule):
    """
    CAREamics Lightning module.

    This class encapsulates the a PyTorch model along with the training, validation,
    and testing logic. It is configured using an `AlgorithmModel` Pydantic class.

    Parameters
    ----------
    algorithm_config : Union[AlgorithmModel, dict]
        Algorithm configuration.

    Attributes
    ----------
    model : nn.Module
        PyTorch model.
    loss_func : nn.Module
        Loss function.
    optimizer_name : str
        Optimizer name.
    optimizer_params : dict
        Optimizer parameters.
    lr_scheduler_name : str
        Learning rate scheduler name.
    """

    def __init__(self, algorithm_config: Union[AlgorithmConfig, dict]) -> None:
        """Lightning module for CAREamics.

        This class encapsulates the a PyTorch model along with the training, validation,
        and testing logic. It is configured using an `AlgorithmModel` Pydantic class.

        Parameters
        ----------
        algorithm_config : Union[AlgorithmModel, dict]
            Algorithm configuration.
        """
        super().__init__()
        # if loading from a checkpoint, AlgorithmModel needs to be instantiated
        if isinstance(algorithm_config, dict):
            algorithm_config = AlgorithmConfig(**algorithm_config)

        # create model and loss function
        self.model: nn.Module = model_factory(algorithm_config.model)
        self.loss_func = loss_factory(algorithm_config.loss)

        # save optimizer and lr_scheduler names and parameters
        self.optimizer_name = algorithm_config.optimizer.name
        self.optimizer_params = algorithm_config.optimizer.parameters
        self.lr_scheduler_name = algorithm_config.lr_scheduler.name
        self.lr_scheduler_params = algorithm_config.lr_scheduler.parameters

    def forward(self, x: Any) -> Any:
        """Forward pass.

        Parameters
        ----------
        x : Any
            Input tensor.

        Returns
        -------
        Any
            Output tensor.
        """
        return self.model(x)

    def training_step(self, batch: Tensor, batch_idx: Any) -> Any:
        """Training step.

        Parameters
        ----------
        batch : Tensor
            Input batch.
        batch_idx : Any
            Batch index.

        Returns
        -------
        Any
            Loss value.
        """
        x, *aux = batch
        out = self.model(x)
        loss = self.loss_func(out, *aux)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch: Tensor, batch_idx: Any) -> None:
        """Validation step.

        Parameters
        ----------
        batch : Tensor
            Input batch.
        batch_idx : Any
            Batch index.
        """
        x, *aux = batch
        out = self.model(x)
        val_loss = self.loss_func(out, *aux)

        # log validation loss
        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def predict_step(self, batch: Tensor, batch_idx: Any) -> Any:
        """Prediction step.

        Parameters
        ----------
        batch : Tensor
            Input batch.
        batch_idx : Any
            Batch index.

        Returns
        -------
        Any
            Model output.
        """
        if self._trainer.datamodule.tiled:
            x, *aux = batch
        else:
            x = batch
            aux = []

        # apply test-time augmentation if available
        # TODO: probably wont work with batch size > 1
        if self._trainer.datamodule.prediction_config.tta_transforms:
            tta = ImageRestorationTTA()
            augmented_batch = tta.forward(x)  # list of augmented tensors
            augmented_output = []
            for augmented in augmented_batch:
                augmented_pred = self.model(augmented)
                augmented_output.append(augmented_pred)
            output = tta.backward(augmented_output)
        else:
            output = self.model(x)

        # Denormalize the output
        denorm = Denormalize(
            image_means=self._trainer.datamodule.predict_dataset.image_means,
            image_stds=self._trainer.datamodule.predict_dataset.image_stds,
        )
        denormalized_output = denorm(patch=output.cpu().numpy())

        if len(aux) > 0:  # aux can be tiling information
            return denormalized_output, *aux
        else:
            return denormalized_output

    def configure_optimizers(self) -> Any:
        """Configure optimizers and learning rate schedulers.

        Returns
        -------
        Any
            Optimizer and learning rate scheduler.
        """
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


class CAREamicsModuleWrapper(CAREamicsModule):
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
        model_parameters: Optional[dict] = None,
        optimizer: Union[SupportedOptimizer, str] = "Adam",
        optimizer_parameters: Optional[dict] = None,
        lr_scheduler: Union[SupportedScheduler, str] = "ReduceLROnPlateau",
        lr_scheduler_parameters: Optional[dict] = None,
    ) -> None:
        """
        Wrapper for the CAREamics model, exposing all algorithm configuration arguments.

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
        # create a AlgorithmModel compatible dictionary
        if lr_scheduler_parameters is None:
            lr_scheduler_parameters = {}
        if optimizer_parameters is None:
            optimizer_parameters = {}
        if model_parameters is None:
            model_parameters = {}
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
        super().__init__(AlgorithmConfig(**algorithm_configuration))

        # TODO add load_from_checkpoint wrapper
