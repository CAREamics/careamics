"""CARE Lightning Module."""

from collections.abc import Callable
from typing import Any

import pytorch_lightning as L
import torch
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio

from careamics.config import CAREAlgorithm, N2NAlgorithm, algorithm_factory
from careamics.config.support import SupportedLoss
from careamics.dataset_ng.dataset import ImageRegionData
from careamics.losses import mae_loss, mse_loss
from careamics.models.unet import UNet
from careamics.utils.logging import get_logger
from careamics.utils.torch_utils import get_optimizer, get_scheduler

from .module_utils import load_best_checkpoint, log_training_stats, log_validation_stats

logger = get_logger(__name__)


class CAREModule(L.LightningModule):
    """CAREamics PyTorch Lightning module for CARE algorithm.

    Parameters
    ----------
    algorithm_config : CAREAlgorithm, N2NAlgorithm, or dict
        Configuration for the CARE algorithm, either as a CAREAlgorithm/N2NAlgorithm
        instance or a dictionary.
    """

    def __init__(self, algorithm_config: CAREAlgorithm | N2NAlgorithm | dict) -> None:
        """Instantiate CARE Module.

        Parameters
        ----------
        algorithm_config : CAREAlgorithm, N2NAlgorithm, or dict
            Configuration for the CARE algorithm, either as a CAREAlgorithm/N2NAlgorithm
            instance or a dictionary.
        """
        super().__init__()

        if isinstance(algorithm_config, dict):
            algorithm_config = algorithm_factory(algorithm_config)
        if not isinstance(algorithm_config, CAREAlgorithm | N2NAlgorithm):
            raise TypeError(
                "algorithm_config must be a CAREAlgorithm or a N2NAlgorithm"
            )

        self.config = algorithm_config
        self.model: nn.Module = UNet(**algorithm_config.model.model_dump())
        loss = algorithm_config.loss
        if loss == SupportedLoss.MAE:
            self.loss_func: Callable = mae_loss
        elif loss == SupportedLoss.MSE:
            self.loss_func = mse_loss
        else:
            raise ValueError(f"Unsupported loss for Care: {loss}")

        self.metrics = MetricCollection(PeakSignalNoiseRatio())

        self._best_checkpoint_loaded = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output tensor.
        """
        return self.model(x)

    def training_step(
        self,
        batch: tuple[ImageRegionData, ImageRegionData],
        batch_idx: int,
    ) -> torch.Tensor:
        """Training step for CARE module.

        Parameters
        ----------
        batch : (ImageRegionData, ImageRegionData)
            A tuple containing the input data and the target data.
        batch_idx : int
            The index of the current batch in the training loop.

        Returns
        -------
        torch.Tensor
            The loss value computed for the current batch.
        """
        # TODO: add validation to determine if target is initialized
        x, target = batch[0], batch[1]

        prediction = self.model(x.data)
        loss = self.loss_func(prediction, target.data)

        log_training_stats(self, loss, batch_size=x.data.shape[0])

        return loss

    def validation_step(
        self,
        batch: tuple[ImageRegionData, ImageRegionData],
        batch_idx: int,
    ) -> None:
        """Validation step for CARE module.

        Parameters
        ----------
        batch : (ImageRegionData, ImageRegionData)
            A tuple containing the input data and the target data.
        batch_idx : int
            The index of the current batch in the validation loop.
        """
        x, target = batch[0], batch[1]

        prediction = self.model(x.data)
        val_loss = self.loss_func(prediction, target.data)
        self.metrics(prediction, target.data)
        log_validation_stats(
            self, val_loss, batch_size=x.data.shape[0], metrics=self.metrics
        )

    def predict_step(
        self,
        batch: tuple[ImageRegionData] | tuple[ImageRegionData, ImageRegionData],
        batch_idx: int,
        load_best_ckpt: bool = False,
    ) -> ImageRegionData:
        """Prediction step for CARE module.

        Parameters
        ----------
        batch : ImageRegionData or (ImageRegionData, ImageRegionData)
            A tuple containing the input data and optionally the target data.
        batch_idx : int
            The index of the current batch in the prediction loop.
        load_best_ckpt : bool, default=False
            Whether to load the best checkpoint before making predictions.

        Returns
        -------
        ImageRegionData
            The output batch containing the predictions.
        """
        if not self._best_checkpoint_loaded and load_best_ckpt:
            self._best_checkpoint_loaded = load_best_checkpoint(self)

        x = batch[0]
        # TODO: add TTA
        prediction = self.model(x.data)

        normalization = self._trainer.datamodule.predict_dataset.normalization
        denormalized_output = normalization.denormalize(prediction).cpu().numpy()

        output_batch = ImageRegionData(
            data=denormalized_output,
            source=x.source,
            data_shape=x.data_shape,
            dtype=x.dtype,
            axes=x.axes,
            region_spec=x.region_spec,
            additional_metadata={},
        )
        return output_batch

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and learning rate scheduler.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the optimizer and learning rate scheduler.
        """
        optimizer_func = get_optimizer(self.config.optimizer.name)
        optimizer = optimizer_func(
            self.model.parameters(), **self.config.optimizer.parameters
        )

        scheduler_func = get_scheduler(self.config.lr_scheduler.name)
        scheduler = scheduler_func(optimizer, **self.config.lr_scheduler.parameters)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
