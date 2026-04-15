"""Deprecated CAREamics Lightning module."""

from typing import Any, Literal, Union

import numpy as np
import pytorch_lightning as L
import torch

from careamics.compat.config.configuration_factories import algorithm_factory
from careamics.compat.config.data.tile_information import TileInformation
from careamics.compat.losses import loss_factory
from careamics.compat.transforms.normalize import Denormalize, TrainDenormalize
from careamics.compat.transforms.tta import ImageRestorationTTA
from careamics.config import (
    N2VAlgorithm,
    PN2VAlgorithm,
    UNetBasedAlgorithm,
)
from careamics.config.support import (
    SupportedAlgorithm,
    SupportedArchitecture,
    SupportedLoss,
    SupportedOptimizer,
    SupportedScheduler,
)
from careamics.lightning.lightning_modules.module_utils import (
    get_optimizer,
    get_scheduler,
)
from careamics.lightning.lightning_modules.n2v_utils import N2VManipulate
from careamics.models.lvae.noise_models import (
    GaussianMixtureNoiseModel,
    MultiChannelNoiseModel,
    noise_model_factory,
)
from careamics.models.model_factory import model_factory

NoiseModel = Union[GaussianMixtureNoiseModel, MultiChannelNoiseModel]


class FCNModule(L.LightningModule):
    """
    CAREamics Lightning module.

    This class encapsulates the PyTorch model along with the training, validation,
    and testing logic. It is configured using an `AlgorithmModel` Pydantic class.

    Parameters
    ----------
    algorithm_config : AlgorithmModel or dict
        Algorithm configuration.

    Attributes
    ----------
    model : torch.nn.Module
        PyTorch model.
    loss_func : torch.nn.Module
        Loss function.
    optimizer_name : str
        Optimizer name.
    optimizer_params : dict
        Optimizer parameters.
    lr_scheduler_name : str
        Learning rate scheduler name.
    """

    def __init__(self, algorithm_config: Union[UNetBasedAlgorithm, dict]) -> None:
        """Lightning module for CAREamics.

        This class encapsulates the a PyTorch model along with the training, validation,
        and testing logic. It is configured using an `AlgorithmModel` Pydantic class.

        Parameters
        ----------
        algorithm_config : AlgorithmModel or dict
            Algorithm configuration.
        """
        super().__init__()

        if isinstance(algorithm_config, dict):
            algorithm_config = algorithm_factory(algorithm_config)

        self.algorithm_config = algorithm_config
        # create preprocessing, model and loss function
        if isinstance(self.algorithm_config, N2VAlgorithm | PN2VAlgorithm):
            self.use_n2v = True
            self.n2v_preprocess: N2VManipulate | None = N2VManipulate(
                self.algorithm_config.n2v_config
            )
        else:
            self.use_n2v = False
            self.n2v_preprocess = None

        self.algorithm = self.algorithm_config.algorithm
        self.model: torch.nn.Module = model_factory(self.algorithm_config.model)
        self.noise_model: NoiseModel | None = noise_model_factory(
            self.algorithm_config.noise_model
            if isinstance(self.algorithm_config, PN2VAlgorithm)
            else None
        )

        # Create loss function, pre-configure with noise model for PN2V
        loss_func = loss_factory(self.algorithm_config.loss)
        if (
            isinstance(self.algorithm_config, PN2VAlgorithm)
            and self.noise_model is not None
        ):
            # For PN2V, reorder arguments and pass noise model
            self.loss_func = lambda *args: loss_func(
                args[0], args[1], args[2], self.noise_model
            )
        else:
            self.loss_func = loss_func

        # save optimizer and lr_scheduler names and parameters
        self.optimizer_name = self.algorithm_config.optimizer.name
        self.optimizer_params = self.algorithm_config.optimizer.parameters
        self.lr_scheduler_name = self.algorithm_config.lr_scheduler.name
        self.lr_scheduler_params = self.algorithm_config.lr_scheduler.parameters

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

    def _train_denormalize(self, out: torch.Tensor) -> torch.Tensor:
        """Denormalize output using training dataset statistics.

        Parameters
        ----------
        out : torch.Tensor
            Output tensor to denormalize.

        Returns
        -------
        torch.Tensor
            Denormalized tensor.
        """
        denorm = TrainDenormalize(
            image_means=(self._trainer.datamodule.train_dataset.image_stats.means),
            image_stds=(self._trainer.datamodule.train_dataset.image_stats.stds),
        )
        return denorm(patch=out)

    def _predict_denormalize(
        self, out: torch.Tensor, from_prediction: bool
    ) -> torch.Tensor:
        """Denormalize output for prediction.

        Parameters
        ----------
        out : torch.Tensor
            Output tensor to denormalize.
        from_prediction : bool
            Whether using prediction or training dataset stats.

        Returns
        -------
        torch.Tensor
            Denormalized tensor.
        """
        denorm = Denormalize(
            image_means=(
                self._trainer.datamodule.predict_dataset.image_means
                if from_prediction
                else self._trainer.datamodule.train_dataset.image_stats.means
            ),
            image_stds=(
                self._trainer.datamodule.predict_dataset.image_stds
                if from_prediction
                else self._trainer.datamodule.train_dataset.image_stats.stds
            ),
        )
        return denorm(patch=out.cpu().numpy())

    def training_step(self, batch: torch.Tensor, batch_idx: Any) -> Any:
        """Training step.

        Parameters
        ----------
        batch : torch.torch.Tensor
            Input batch.
        batch_idx : Any
            Batch index.

        Returns
        -------
        Any
            Loss value.
        """
        x, *targets = batch
        if self.use_n2v and self.n2v_preprocess is not None:
            x_preprocessed, *aux = self.n2v_preprocess(x)
        else:
            x_preprocessed = x
            aux = []

        out = self.model(x_preprocessed)

        # PN2V needs denormalized output and targets for loss computation
        if isinstance(self.algorithm_config, PN2VAlgorithm):
            out = self._train_denormalize(out)
            aux = [self._train_denormalize(aux[0]), aux[1]]
            # TODO hacky and ugly
        loss = self.loss_func(out, *aux, *targets)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        optimizer = self.optimizers()
        current_lr = optimizer.param_groups[0]["lr"]
        self.log("learning_rate", current_lr, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: Any) -> None:
        """Validation step.

        Parameters
        ----------
        batch : torch.torch.Tensor
            Input batch.
        batch_idx : Any
            Batch index.
        """
        x, *targets = batch
        if self.use_n2v and self.n2v_preprocess is not None:
            x_preprocessed, *aux = self.n2v_preprocess(x)
        else:
            x_preprocessed = x
            aux = []

        out = self.model(x_preprocessed)

        # PN2V needs denormalized output and targets for loss computation
        if isinstance(self.algorithm_config, PN2VAlgorithm):
            out = torch.tensor(self._train_denormalize(out))
            aux = [self._train_denormalize(aux[0]), aux[1]]
            # TODO hacky and ugly
        val_loss = self.loss_func(out, *aux, *targets)

        # log validation loss
        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def predict_step(self, batch: torch.Tensor, batch_idx: Any) -> Any:
        """Prediction step.

        Parameters
        ----------
        batch : torch.torch.torch.Tensor
            Input batch.
        batch_idx : Any
            Batch index.

        Returns
        -------
        Any
            Model output.
        """
        # TODO refactor when redoing datasets
        # hacky way to determine if it is PredictDataModule, otherwise there is a
        # circular import to solve with isinstance
        from_prediction = hasattr(self._trainer.datamodule, "tiled")
        is_tiled = (
            len(batch) > 1
            and isinstance(batch[1], list)
            and isinstance(batch[1][0], TileInformation)
        )

        # TODO add explanations for what is happening here
        if is_tiled:
            x, *aux = batch
            if type(x) in [list, tuple]:
                x = x[0]
        else:
            if type(batch) in [list, tuple]:
                x = batch[0]  # TODO change, ugly way to deal with n2v refac
            else:
                x = batch
            aux = []

        # apply test-time augmentation if available
        # TODO: probably wont work with batch size > 1
        if (
            from_prediction
            and self._trainer.datamodule.prediction_config.tta_transforms
        ):
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
        # TODO incompatible API between predict and train datasets

        denormalized_input = self._predict_denormalize(
            x, from_prediction=from_prediction
        )
        denormalized_output = self._predict_denormalize(
            output, from_prediction=from_prediction
        )

        # Calculate MSE estimate
        if isinstance(self.algorithm_config, PN2VAlgorithm):
            assert self.noise_model is not None, "Noise model required for PN2V"
            likelihoods = self.noise_model.likelihood(
                torch.tensor(denormalized_input), torch.tensor(denormalized_output)
            )
            mse_estimate = torch.sum(
                likelihoods * denormalized_output, dim=1, keepdim=True
            )
            mse_estimate /= torch.sum(likelihoods, dim=1, keepdim=True)

        if isinstance(self.algorithm_config, PN2VAlgorithm):
            denormalized_output = np.mean(denormalized_output, axis=1, keepdims=True)
            denormalized_output = (denormalized_output, mse_estimate)
            # TODO: might be ugly but otherwise we need to change the output signature
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


# TODO: make this LVAE compatible (?)
def create_careamics_module(
    algorithm: Union[SupportedAlgorithm, str],
    loss: Union[SupportedLoss, str],
    architecture: Union[SupportedArchitecture, str],
    use_n2v2: bool = False,
    struct_n2v_axis: Literal["horizontal", "vertical", "none"] = "none",
    struct_n2v_span: int = 5,
    model_parameters: dict | None = None,
    optimizer: Union[SupportedOptimizer, str] = "Adam",
    optimizer_parameters: dict | None = None,
    lr_scheduler: Union[SupportedScheduler, str] = "ReduceLROnPlateau",
    lr_scheduler_parameters: dict | None = None,
) -> FCNModule:
    """Create a CAREamics Lightning module.

    This function exposes parameters used to create an AlgorithmModel instance,
    triggering parameters validation.

    Parameters
    ----------
    algorithm : SupportedAlgorithm or str
        Algorithm to use for training (see SupportedAlgorithm).
    loss : SupportedLoss or str
        Loss function to use for training (see SupportedLoss).
    architecture : SupportedArchitecture or str
        Model architecture to use for training (see SupportedArchitecture).
    use_n2v2 : bool, default=False
        Whether to use N2V2 or Noise2Void.
    struct_n2v_axis : "horizontal", "vertical", or "none", default="none"
        Axis of the StructN2V mask.
    struct_n2v_span : int, default=5
        Span of the StructN2V mask.
    model_parameters : dict, optional
        Model parameters to use for training, by default {}. Model parameters are
        defined in the relevant `torch.nn.Module` class, or Pyddantic model (see
        `careamics.config.architectures`).
    optimizer : SupportedOptimizer or str, optional
        Optimizer to use for training, by default "Adam" (see SupportedOptimizer).
    optimizer_parameters : dict, optional
        Optimizer parameters to use for training, as defined in `torch.optim`, by
        default {}.
    lr_scheduler : SupportedScheduler or str, optional
        Learning rate scheduler to use for training, by default "ReduceLROnPlateau"
        (see SupportedScheduler).
    lr_scheduler_parameters : dict, optional
        Learning rate scheduler parameters to use for training, as defined in
        `torch.optim`, by default {}.

    Returns
    -------
    CAREamicsModule
        CAREamics Lightning module.
    """
    # TODO should use the same functions are in configuration_factory.py
    # create an AlgorithmModel compatible dictionary
    if lr_scheduler_parameters is None:
        lr_scheduler_parameters = {}
    if optimizer_parameters is None:
        optimizer_parameters = {}
    if model_parameters is None:
        model_parameters = {}
    algorithm_dict: dict[str, Any] = {
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

    model_dict = {"architecture": architecture}
    model_dict.update(model_parameters)

    # add model parameters to algorithm configuration
    algorithm_dict["model"] = model_dict

    which_algo = algorithm_dict["algorithm"]
    if which_algo in UNetBasedAlgorithm.get_compatible_algorithms():
        algorithm_cfg = algorithm_factory(algorithm_dict)

        # if use N2V
        if isinstance(algorithm_cfg, N2VAlgorithm | PN2VAlgorithm):
            algorithm_cfg.n2v_config.struct_mask_axis = struct_n2v_axis
            algorithm_cfg.n2v_config.struct_mask_span = struct_n2v_span
            algorithm_cfg.set_n2v2(use_n2v2)

        return FCNModule(algorithm_cfg)
    else:
        raise NotImplementedError(
            f"Algorithm {which_algo} is not implemented or unknown."
        )
