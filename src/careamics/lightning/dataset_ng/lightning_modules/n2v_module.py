"""Noise2Void Lightning Module."""

from typing import Any, cast

import pytorch_lightning as L
import torch
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio

from careamics.config import N2VAlgorithm, algorithm_factory
from careamics.dataset_ng.dataset import ImageRegionData
from careamics.losses import n2v_loss
from careamics.models.unet import UNet
from careamics.transforms import N2VManipulateTorch
from careamics.utils.logging import get_logger

from .module_utils import configure_optimizers, log_training_stats, log_validation_stats

logger = get_logger(__name__)


class N2VModule(L.LightningModule):
    """CAREamics PyTorch Lightning module for N2V algorithm.

    Parameters
    ----------
    algorithm_config : N2VAlgorithm or dict
        Configuration for the N2V algorithm, either as an N2VAlgorithm instance or a
        dictionary.
    """

    def __init__(self, algorithm_config: N2VAlgorithm | dict[str, Any]) -> None:
        """Instantiate N2VModule.

        Parameters
        ----------
        algorithm_config : N2VAlgorithm or dict
            Configuration for the N2V algorithm, either as an N2VAlgorithm instance or a
            dictionary.
        """
        super().__init__()

        if isinstance(algorithm_config, dict):
            config = algorithm_factory(algorithm_config)
        else:
            config = algorithm_config

        if not isinstance(config, N2VAlgorithm):
            raise TypeError("algorithm_config must be a N2VAlgorithm")

        self.save_hyperparameters({"algorithm_config": config.model_dump(mode="json")})
        self.config = config
        self.model: nn.Module = UNet(**self.config.model.model_dump())
        self.n2v_manipulate = N2VManipulateTorch(self.config.n2v_config)
        self.loss_func = n2v_loss

        self.metrics = MetricCollection(PeakSignalNoiseRatio())

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
        batch: tuple[ImageRegionData] | tuple[ImageRegionData, ImageRegionData],
        batch_idx: int,
    ) -> torch.Tensor:
        """Training step for N2V model.

        Parameters
        ----------
        batch : ImageRegionData or (ImageRegionData, ImageRegionData)
            A tuple containing the input data and the target data.
        batch_idx : int
            The index of the current batch in the training loop.

        Returns
        -------
        torch.Tensor
            The loss value for the current training step.
        """
        x = batch[0]
        x_data = cast(torch.Tensor, x.data)
        x_masked, x_original, mask = self.n2v_manipulate(x_data)
        prediction = self.model(x_masked)
        loss = self.loss_func(prediction, x_original, mask)

        log_training_stats(self, loss, batch_size=x_data.shape[0])

        return loss

    def validation_step(
        self,
        batch: tuple[ImageRegionData] | tuple[ImageRegionData, ImageRegionData],
        batch_idx: int,
    ) -> None:
        """Validation step for N2V model.

        Parameters
        ----------
        batch : ImageRegionData or (ImageRegionData, ImageRegionData)
            A tuple containing the input data and the target data.
        batch_idx : int
            The index of the current batch in the validation loop.
        """
        x = batch[0]
        x_data = cast(torch.Tensor, x.data)
        x_masked, x_original, mask = self.n2v_manipulate(x_data)
        prediction = self.model(x_masked)
        val_loss = self.loss_func(prediction, x_original, mask)
        self.metrics(prediction, x_original)
        log_validation_stats(
            self, val_loss, batch_size=x_data.shape[0], metrics=self.metrics
        )

    def predict_step(
        self,
        batch: tuple[ImageRegionData] | tuple[ImageRegionData, ImageRegionData],
        batch_idx: int,
    ) -> ImageRegionData:
        """Prediction step for N2V model.

        Parameters
        ----------
        batch : ImageRegionData or (ImageRegionData, ImageRegionData)
            A tuple containing the input data and optionally the target data.
        batch_idx : int
            The index of the current batch in the prediction loop.

        Returns
        -------
        ImageRegionData
            The output batch containing the predictions.
        """
        x = batch[0]
        x_data = cast(torch.Tensor, x.data)
        # TODO: add TTA
        prediction = self.model(x_data)

        normalization = self._trainer.datamodule.predict_dataset.normalization  # type: ignore[union-attr]
        denormalized_output = normalization.denormalize(prediction).cpu().numpy()

        output_batch = ImageRegionData(
            data=denormalized_output,
            source=x.source,
            data_shape=x.data_shape,
            dtype=x.dtype,
            axes=x.axes,
            region_spec=x.region_spec,
            additional_metadata={},
            original_data_shape=x.original_data_shape,
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
