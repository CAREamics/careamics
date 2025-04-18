from typing import Any, Union, Optional

import pytorch_lightning as L
import torch
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio

from careamics.config import (
    N2VAlgorithm,
    UNetBasedAlgorithm,
    algorithm_factory,
)
from careamics.dataset_ng.dataset import ImageRegionData
from careamics.losses import loss_factory
from careamics.models.model_factory import model_factory
from careamics.transforms import N2VManipulateTorch
from careamics.transforms.normalize import Denormalize
from careamics.utils.logging import get_logger
from careamics.utils.torch_utils import get_optimizer, get_scheduler

logger = get_logger(__name__)


class UNetModule(L.LightningModule):
    def __init__(self, algorithm_config: Union[UNetBasedAlgorithm, dict]) -> None:
        super().__init__()

        if isinstance(algorithm_config, dict):
            algorithm_config = algorithm_factory(algorithm_config)

        self.config = algorithm_config

        self.model: nn.Module = model_factory(algorithm_config.model)
        self.loss_func = loss_factory(algorithm_config.loss)

        if isinstance(algorithm_config, N2VAlgorithm):
            self.preprocessing: Optional[N2VManipulateTorch] = N2VManipulateTorch(
                n2v_manipulate_config=algorithm_config.n2v_config
            )
        else:
            self.preprocessing = None

        self.save_hyperparameters({"algorithm_config": algorithm_config.model_dump()})

        self._best_checkpoint_loaded = False

        # TODO: how to support metric evaluation better
        self.metrics = MetricCollection(PeakSignalNoiseRatio())

    def forward(self, x: Any) -> Any:
        return self.model(x)

    def training_step(
        self,
        batch: Union[tuple[ImageRegionData], tuple[ImageRegionData, ImageRegionData]],
        batch_idx: Any,
    ) -> Any:
        if len(batch) > 1:
            x, target = batch[0], batch[1]
        else:
            x = batch[0]
            target = None

        if self.preprocessing is not None:
            x_masked, x_original, mask = self.preprocessing(x.data)
            prediction = self.model(x_masked)
            loss_args = [x_original, mask]
        else:
            prediction = self.model(x.data)
            loss_args = []

        if target is not None:
            loss_args.append(target.data)

        loss = self.loss_func(prediction, *loss_args)

        batch_size = self._trainer.datamodule.config.batch_size
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
        current_lr = optimizer.param_groups[0]["lr"]
        self.log(
            "learning_rate",
            current_lr,
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
        )

        return loss

    def validation_step(
        self,
        batch: Union[tuple[ImageRegionData], tuple[ImageRegionData, ImageRegionData]],
        batch_idx: Any,
    ) -> None:
        if len(batch) > 1:
            x, target = batch[0], batch[1]
        else:
            x = batch[0]
            target = None

        if self.preprocessing is not None:
            x_masked, x_original, mask = self.preprocessing(x.data)
            prediction = self.model(x_masked)
            loss_args = [x_original, mask]
        else:
            prediction = self.model(x.data)
            loss_args = []

        if target is not None:
            loss_args.append(target.data)

        val_loss = self.loss_func(prediction, *loss_args)

        batch_size = self._trainer.datamodule.config.batch_size
        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )

        if target is not None:
            self.metrics(prediction, target.data)
        elif self.preprocessing is not None:
            self.metrics(prediction, x_original)
        else:
            self.metrics(prediction, x.data)

        self.log_dict(self.metrics, on_step=False, on_epoch=True, batch_size=batch_size)

        if batch_idx == 0:
            self.logger.log_image(images=[x.data], key="input_images")
            self.logger.log_image(images=[prediction], key="predicted_images")

            # TODO: check if it works with other loggers
            if target is not None:
                self.logger.log_image(images=[target.data], key="target_images")

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
    ) -> Any:
        if batch_idx == 0 and self._best_checkpoint_loaded is False:
            self._load_best_checkpoint()
            self._best_checkpoint_loaded = True

        x = batch[0]
        prediction = self.model(x.data).cpu().numpy()

        means = self._trainer.datamodule.stats.means
        stds = self._trainer.datamodule.stats.stds
        denormalize = Denormalize(
            image_means=means,
            image_stds=stds,
        )
        denormalized_output = denormalize(prediction)

        # TODO: add TTA

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
