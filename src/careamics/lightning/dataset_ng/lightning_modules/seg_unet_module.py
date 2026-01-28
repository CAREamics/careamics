"""UNet-based segmentation Lightning Module."""

from typing import Any

import pytorch_lightning as L
import torch
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.segmentation import GeneralizedDiceScore

from careamics.config import SegAlgorithm, algorithm_factory
from careamics.dataset_ng.dataset import ImageRegionData
from careamics.lightning.dataset_ng.loss import get_seg_loss
from careamics.models.unet import UNet
from careamics.utils.logging import get_logger

from .module_utils import (
    configure_optimizers,
    load_best_checkpoint,
    log_training_stats,
    log_validation_stats,
)

logger = get_logger(__name__)


class SegModule(L.LightningModule):
    """CAREamics PyTorch Lightning module for UNet-based segmentation.

    Parameters
    ----------
    algorithm_config : SegAlgorithm or dict
        Configuration for the segmentation algorithm, either as a SegAlgorithm
        instance or a dictionary.
    """

    def __init__(self, algorithm_config: SegAlgorithm | dict) -> None:
        """Instantiate Segmentation Module.

        Parameters
        ----------
        algorithm_config : SegAlgorithm or dict
            Configuration for the segmentation algorithm, either as a SegAlgorithm
            instance or a dictionary.
        """
        super().__init__()

        if isinstance(algorithm_config, dict):
            config = algorithm_factory(algorithm_config)
        else:
            config = algorithm_config

        if not isinstance(config, SegAlgorithm):
            raise TypeError(
                "Parameter `algorithm_config` must be a SegAlgorithm, or a dict "
                "that representing a SegAlgorithm Pydantic model."
            )

        self.config = config
        self.model: nn.Module = UNet(**self.config.model.model_dump())
        loss = self.config.loss
        self.loss_func = get_seg_loss(loss)

        self.metrics: MetricCollection = MetricCollection(
            GeneralizedDiceScore(num_classes=self.config.model.num_classes)
        )

        self._best_checkpoint_loaded: bool = True

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
        """Training step for segmentation module.

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
        """Validation step for segmentation module.

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

        # convert predictions to class indices for metrics
        # for binary (1 channel): apply sigmoid and threshold
        # for multi-class (>1 channels): apply argmax
        if prediction.shape[1] == 1:
            pred_classes = (prediction.sigmoid() > 0.5).long()
        else:
            pred_classes = prediction.argmax(dim=1, keepdim=True)

        self.metrics(pred_classes, target.data)
        log_validation_stats(
            self, val_loss, batch_size=x.data.shape[0], metrics=self.metrics
        )

    def predict_step(
        self,
        batch: tuple[ImageRegionData] | tuple[ImageRegionData, ImageRegionData],
        batch_idx: int,
        load_best_ckpt: bool = True,
    ) -> ImageRegionData:
        """Prediction step for segmentation module.

        Parameters
        ----------
        batch : ImageRegionData or (ImageRegionData, ImageRegionData)
            A tuple containing the input data and optionally the target data.
        batch_idx : int
            The index of the current batch in the prediction loop.
        load_best_ckpt : bool, default=True
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

        # apply appropriate activation function based on number of classes
        # for binary (1 channel): apply sigmoid to get probabilities
        # for multi-class (>1 channels): apply softmax to get class probabilities
        if prediction.shape[1] == 1:
            prediction = prediction.sigmoid()
        else:
            prediction = prediction.softmax(dim=1)

        prediction = prediction.cpu().numpy()

        output_batch = ImageRegionData(
            data=prediction,
            source=x.source,
            data_shape=x.data_shape,
            dtype=x.dtype,
            axes=x.axes,
            region_spec=x.region_spec,
            additional_metadata={},
        )
        return output_batch

    def configure_optimizers(self) -> dict[str, Any]:  # type: ignore[override]
        """Configure optimizer and learning rate scheduler.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the optimizer and learning rate scheduler.
        """
        return configure_optimizers(
            model=self.model,
            optimizer_name=self.config.optimizer.name,
            optimizer_parameters=self.config.optimizer.parameters,
            lr_scheduler_name=self.config.lr_scheduler.name,
            lr_scheduler_parameters=self.config.lr_scheduler.parameters,
        )
