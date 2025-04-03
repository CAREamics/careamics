from typing import Any, Optional, Union

import pytorch_lightning as L
from torch import Tensor, nn

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
from careamics.utils.torch_utils import get_optimizer, get_scheduler


# TODO: go over logging and hyper parameter saving
# TODO: add support for VAE
# TODO: test prediction with tiling
# TODO: move stitching to prediction ??
# TODO: how to pass metrics ??


class N2VModule(L.LightningModule):
    def __init__(self, algorithm_config: Union[UNetBasedAlgorithm, dict]) -> None:
        super().__init__()

        if isinstance(algorithm_config, dict):
            algorithm_config = algorithm_factory(algorithm_config)

        assert isinstance(
            algorithm_config, N2VAlgorithm
        ), "Algorithm config must be a N2VAlgorithm"

        self.config = algorithm_config

        self.model: nn.Module = model_factory(algorithm_config.model)
        self.loss_func = loss_factory(algorithm_config.loss)
        self.preprocessing: N2VManipulateTorch = N2VManipulateTorch(
            n2v_manipulate_config=algorithm_config.n2v_config
        )

    def forward(self, x: Any) -> Any:
        return self.model(x)

    def training_step(self, batch: ImageRegionData, batch_idx: Any) -> Any:
        x, *targets = batch

        x_masked, x_original, mask = self.preprocessing(x.data)
        prediction = self.model(x_masked)

        loss_args = (x_original, mask)
        loss = self.loss_func(prediction, *loss_args, *targets)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch: ImageRegionData, batch_idx: Any) -> None:
        x, *targets = batch

        x_masked, x_original, mask = self.preprocessing(x.data)
        prediction = self.model(x_masked)

        loss_args = (x_original, mask)
        val_loss = self.loss_func(prediction, *loss_args, *targets)

        # log validation loss
        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def predict_step(self, batch: ImageRegionData, batch_idx: Any) -> Any:
        # TODO: add TTA
        x: ImageRegionData = batch[0]
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
            region_spec=x.region_spec
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


class UnetModule(L.LightningModule):
    pass
