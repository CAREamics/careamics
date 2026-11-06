"""CryoCARE Lightning Module."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import lightning as L
import torch
from torch import nn
from torch.nn import L1Loss, MSELoss
from torchmetrics import MetricCollection

from careamics.config import CryoCAREAlgorithm
from careamics.config.factories.config_discriminators import (
    instantiate_algorithm_config,
)
from careamics.config.support import SupportedLoss
from careamics.dataset import ImageRegionData
from careamics.dataset.factory import TrainValData, TrainValSplitData
from careamics.lightning.logger import CoLogger
from careamics.metrics import SIPSNR
from careamics.models.unet import UNet
from careamics.utils.logging import get_logger

from .module_utils import (
    configure_optimizers,
    log_images,
    log_training_stats,
    log_validation_stats,
)

if TYPE_CHECKING:
    from careamics.lightning.data.data_module import CareamicsDataModule

logger = get_logger(__name__)


class CryoCAREModule(L.LightningModule):
    """CAREamics PyTorch Lightning module for cryoCARE.

    Parameters
    ----------
    algorithm_config : CryoCAREAlgorithm or dict
        Configuration for cryoCARE, either as a CryoCAREAlgorithm instance or a
        dictionary.
    """

    def __init__(self, algorithm_config: CryoCAREAlgorithm | dict) -> None:
        """Instantiate cryoCARE Module.

        Parameters
        ----------
        algorithm_config : CryoCAREAlgorithm, or dict
            Configuration for cryoCARE, either as a CryoCAREAlgorithm instance or a
            dictionary.
        """
        super().__init__()

        if isinstance(algorithm_config, dict):
            config = instantiate_algorithm_config(algorithm_config)
        else:
            config = algorithm_config

        if not isinstance(config, CryoCAREAlgorithm):
            raise TypeError("algorithm_config must be a CryoCAREAlgorithm")

        self.save_hyperparameters({"algorithm_config": config.model_dump(mode="json")})
        self.config = config
        self.model: nn.Module = UNet(**self.config.model.model_dump())

        # TODO which loss? use a loss factory
        self.loss_func: Callable = (
            MSELoss() if self.config.loss == SupportedLoss.MSE else L1Loss()
        )

        self.metrics: MetricCollection = MetricCollection(
            {
                f"SIPSNR_{i}": SIPSNR(
                    n_channels=self.config.model.num_classes,
                    output_channel=i,
                    use_scale_invariance=True,
                )
                for i in range(self.config.model.num_classes)
            }
        )

    def on_fit_start(self) -> None:
        """On fit start hook for CARE module.

        Check that training and validation target data have been supplied.
        """
        assert self._trainer is not None
        datamodule: CareamicsDataModule = (
            self._trainer.datamodule  # type: ignore[union-attr]
        )

        # TODO add a cryoCARE typing?
        assert isinstance(datamodule._data, (TrainValData, TrainValSplitData))
        if datamodule._data.train_data_target is None:
            raise ValueError(
                "Training target data must be provided for supervised training."
            )
        if (
            isinstance(datamodule._data, TrainValData)
            and datamodule._data.val_data_target is None
        ):
            raise ValueError(
                "Validation target data must be provided for supervised training."
            )

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
        # log images
        not_sanity_checking = not self.trainer.sanity_checking
        if batch_idx == 0 and isinstance(self.logger, CoLogger) and not_sanity_checking:
            n_samples = min(5, x.data.shape[0])
            log_images(
                module=self,
                input_or_target=target.data[:n_samples],  # type: ignore
                prediction=prediction[:n_samples],
                is_supervised=True,
            )

    def predict_step(
        self,
        batch: tuple[ImageRegionData, ImageRegionData],
        batch_idx: int,
    ) -> ImageRegionData:
        """Predict and average one batch of paired regions.

        Parameters
        ----------
        batch : (ImageRegionData, ImageRegionData)
            Odd and even regions with matching extraction coordinates.
        batch_idx : int
            Index of the prediction batch.

        Returns
        -------
        ImageRegionData
            Averaged, denormalized prediction using the odd/input metadata.
        """
        odd, even = batch
        prediction = 0.5 * (self.model(odd.data) + self.model(even.data))

        normalization = (
            self._trainer.datamodule.predict_dataset.normalization  # type: ignore[union-attr]
        )
        denormalized_output = normalization.denormalize(prediction).cpu().numpy()

        return ImageRegionData.from_model_output(odd, denormalized_output)

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
