from typing import Any, Union

import pytorch_lightning as L
import torch
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio

from careamics.config import algorithm_factory
from careamics.config.algorithms import CAREAlgorithm, N2NAlgorithm, N2VAlgorithm
from careamics.dataset_ng.dataset import ImageRegionData
from careamics.models.unet import UNet
from careamics.transforms import Denormalize
from careamics.utils.logging import get_logger
from careamics.utils.torch_utils import get_optimizer, get_scheduler

logger = get_logger(__name__)


class UnetModule(L.LightningModule):
    """CAREamics PyTorch Lightning module for UNet based algorithms."""

    def __init__(
        self, algorithm_config: Union[CAREAlgorithm, N2VAlgorithm, N2NAlgorithm, dict]
    ) -> None:
        super().__init__()

        if isinstance(algorithm_config, dict):
            algorithm_config = algorithm_factory(algorithm_config)

        self.config = algorithm_config
        self.model: nn.Module = UNet(**algorithm_config.model.model_dump())

        self._best_checkpoint_loaded = False

        # TODO: how to support metric evaluation better
        self.metrics = MetricCollection(PeakSignalNoiseRatio())

    def forward(self, x: Any) -> Any:
        """Default forward method."""
        return self.model(x)

    def _log_training_stats(self, loss: Any, batch_size: Any) -> None:
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )

        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            current_lr = optimizer[0].param_groups[0]["lr"]
        else:
            current_lr = optimizer.param_groups[0]["lr"]
        self.log(
            "learning_rate",
            current_lr,
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
        )

    def _log_validation_stats(self, loss: Any, batch_size: Any) -> None:
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        self.log_dict(self.metrics, on_step=False, on_epoch=True, batch_size=batch_size)

    def _load_best_checkpoint(self) -> None:
        if (
            not hasattr(self.trainer, "checkpoint_callback")
            or self.trainer.checkpoint_callback is None
        ):
            logger.warning("No checkpoint callback found, cannot load best checkpoint.")
            return

        best_model_path = self.trainer.checkpoint_callback.best_model_path
        if best_model_path and best_model_path != "":
            logger.info(f"Loading best checkpoint from: {best_model_path}")
            model_state = torch.load(best_model_path, weights_only=True)["state_dict"]
            self.load_state_dict(model_state)
        else:
            logger.warning("No best checkpoint found.")

    def predict_step(
        self,
        batch: Union[tuple[ImageRegionData], tuple[ImageRegionData, ImageRegionData]],
        batch_idx: Any,
        load_best_checkpoint=False,
    ) -> Any:
        """Default predict step."""
        if self._best_checkpoint_loaded is False and load_best_checkpoint:
            self._load_best_checkpoint()
            self._best_checkpoint_loaded = True

        x = batch[0]
        # TODO: add TTA
        prediction = self.model(x.data).cpu().numpy()

        means = self._trainer.datamodule.stats.means
        stds = self._trainer.datamodule.stats.stds
        denormalize = Denormalize(
            image_means=means,
            image_stds=stds,
        )
        denormalized_output = denormalize(prediction)

        output_batch = ImageRegionData(
            data=denormalized_output,
            source=x.source,
            data_shape=x.data_shape,
            dtype=x.dtype,
            axes=x.axes,
            region_spec=x.region_spec,
        )
        return output_batch

    def configure_optimizers(self) -> Any:
        """Configure optimizers."""
        optimizer_func = get_optimizer(self.config.optimizer.name)
        optimizer = optimizer_func(
            self.model.parameters(), **self.config.optimizer.parameters
        )

        scheduler_func = get_scheduler(self.config.lr_scheduler.name)
        scheduler = scheduler_func(optimizer, **self.config.lr_scheduler.parameters)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",  # otherwise triggers MisconfigurationException
        }
