"""Probabilistic Noise2Void Lightning module."""

from typing import TYPE_CHECKING, Any, cast

import lightning as L
import torch
from torch import nn
from torchmetrics import MetricCollection

from careamics.config import PN2VAlgorithm
from careamics.config.factories.algorithm_factory import algorithm_factory
from careamics.dataset import ImageRegionData
from careamics.dataset.factory import TrainValData, TrainValSplitData
from careamics.dataset.normalization.normalization import Normalization
from careamics.losses import pn2v_loss
from careamics.metrics import SIPSNR
from careamics.models.lvae.noise_models import GaussianMixtureNoiseModel
from careamics.models.unet import UNet
from careamics.utils.logging import get_logger

from .module_utils import configure_optimizers, log_training_stats, log_validation_stats
from .n2v_utils import N2VManipulate

logger = get_logger(__name__)

if TYPE_CHECKING:
    from careamics.lightning.data.data_module import CareamicsDataModule


def pn2v_mmse(
    samples: torch.Tensor,
    observations: torch.Tensor,
    noise_model: GaussianMixtureNoiseModel,
) -> torch.Tensor:
    """Likelihood-weighted mean of prior samples.

    ``samples`` is ``(batch, K, ...)`` and ``observations`` broadcasts over the sample
    axis. The result is the MMSE estimate used for the reported denoised image.

    Parameters
    ----------
    samples : torch.Tensor
        Denormalized prior samples, shape ``(batch, K, ...)``.
    observations : torch.Tensor
        Denormalized noisy observations, shape ``(batch, 1, ...)``.
    noise_model : GaussianMixtureNoiseModel
        Noise model used to evaluate ``p(observation | sample)``.

    Returns
    -------
    torch.Tensor
        MMSE estimate, shape ``(batch, 1, ...)``.
    """
    likelihoods = noise_model.likelihood(observations, samples)
    return torch.sum(likelihoods * samples, dim=1, keepdim=True) / torch.sum(
        likelihoods, dim=1, keepdim=True
    )


class PN2VModule(L.LightningModule):
    """CAREamics PyTorch Lightning module for the PN2V algorithm.

    Parameters
    ----------
    algorithm_config : PN2VAlgorithm or dict
        PN2V algorithm configuration.
    """

    def __init__(self, algorithm_config: PN2VAlgorithm | dict[str, Any]) -> None:
        """Instantiate PN2VModule.

        Parameters
        ----------
        algorithm_config : PN2VAlgorithm or dict
            PN2V algorithm configuration.
        """
        super().__init__()

        if isinstance(algorithm_config, dict):
            config = algorithm_factory(algorithm_config)
        else:
            config = algorithm_config

        if not isinstance(config, PN2VAlgorithm):
            raise TypeError("algorithm_config must be a PN2VAlgorithm")

        self.save_hyperparameters({"algorithm_config": config.model_dump(mode="json")})
        self.config = config
        self.model: nn.Module = UNet(**self.config.model.model_dump())
        self.n2v_manipulate = N2VManipulate(self.config.n2v_config)
        self.noise_model = GaussianMixtureNoiseModel(self.config.noise_model)
        self.loss_func = pn2v_loss

        # One diagnostic metric on the prior mean. Paper PSNR is computed outside
        # this module, on the MMSE prediction.
        self.metrics: MetricCollection = MetricCollection(
            {
                "SIPSNR_0": SIPSNR(
                    n_channels=1,
                    output_channel=0,
                    use_scale_invariance=True,
                )
            }
        )

    def on_fit_start(self) -> None:
        """On fit start hook for the PN2V module."""
        assert self._trainer is not None
        datamodule: CareamicsDataModule = self._trainer.datamodule  # type: ignore[union-attr]
        assert isinstance(datamodule._data, (TrainValData, TrainValSplitData))
        if datamodule._data.train_data_target is not None:
            logger.warning(
                "PN2V is a self-supervised algorithm, arguments passed to "
                "`train_data_target` will be ignored.",
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the UNet.

        The sample scale is applied in the training and prediction steps, not here.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Network output, one channel per prior sample.
        """
        return self.model(x)

    def _normalization(self) -> Normalization:
        """Return the normalization used by the active dataloader.

        Returns
        -------
        Normalization
            Normalization of the dataset for the current trainer stage.

        Raises
        ------
        RuntimeError
            If that dataset has not been set up.
        """
        datamodule = self.trainer.datamodule
        if self.trainer.predicting:
            dataset = datamodule.predict_dataset
        elif self.trainer.validating:
            dataset = datamodule.val_dataset
        else:
            dataset = datamodule.train_dataset
        if dataset is None:
            raise RuntimeError("No dataset is available to denormalize PN2V samples.")
        return dataset.normalization

    def _denormalized_samples(
        self, prediction: torch.Tensor, original: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scale network outputs, then denormalize samples and original pixels.

        Parameters
        ----------
        prediction : torch.Tensor
            Network output in normalized space.
        original : torch.Tensor
            Original noisy patch in normalized space.

        Returns
        -------
        tuple of torch.Tensor
            Denormalized prior samples and denormalized observations.
        """
        normalization = self._normalization()
        samples = normalization.denormalize(prediction * self.config.sample_scale)
        observations = normalization.denormalize(original)
        return samples, observations

    def training_step(
        self,
        batch: tuple[ImageRegionData] | tuple[ImageRegionData, ImageRegionData],
        batch_idx: int,
    ) -> torch.Tensor:
        """Training step for PN2V.

        Parameters
        ----------
        batch : ImageRegionData or (ImageRegionData, ImageRegionData)
            Input batch. A target, when present, is ignored.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            PN2V loss on masked pixels.
        """
        x = batch[0]
        x_data = cast(torch.Tensor, x.data)
        x_masked, x_original, mask = self.n2v_manipulate(x_data)
        prediction = self.model(x_masked)
        samples, observations = self._denormalized_samples(prediction, x_original)
        loss = self.loss_func(samples, observations, mask, self.noise_model)

        log_training_stats(self, loss, batch_size=x_data.shape[0])
        return loss

    def validation_step(
        self,
        batch: tuple[ImageRegionData] | tuple[ImageRegionData, ImageRegionData],
        batch_idx: int,
    ) -> None:
        """Validation step for PN2V.

        Parameters
        ----------
        batch : ImageRegionData or (ImageRegionData, ImageRegionData)
            Input batch. A target, when present, is ignored.
        batch_idx : int
            Index of the current batch.
        """
        x = batch[0]
        x_data = cast(torch.Tensor, x.data)
        x_masked, x_original, mask = self.n2v_manipulate(x_data)
        prediction = self.model(x_masked)
        samples, observations = self._denormalized_samples(prediction, x_original)
        val_loss = self.loss_func(samples, observations, mask, self.noise_model)
        prior_mean = prediction.mean(dim=1, keepdim=True)
        self.metrics(prior_mean, x_original)
        log_validation_stats(
            self, val_loss, batch_size=x_data.shape[0], metrics=self.metrics
        )

    def predict_step(
        self,
        batch: tuple[ImageRegionData] | tuple[ImageRegionData, ImageRegionData],
        batch_idx: int,
    ) -> ImageRegionData:
        """Prediction step.

        The returned image is the MMSE estimate, not the mean of the prior samples.

        Parameters
        ----------
        batch : ImageRegionData or (ImageRegionData, ImageRegionData)
            Input batch.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        ImageRegionData
            MMSE prediction in the original intensity range.
        """
        x = batch[0]
        x_data = cast(torch.Tensor, x.data)
        prediction = self.model(x_data)
        samples, observations = self._denormalized_samples(prediction, x_data)
        mmse = pn2v_mmse(samples, observations, self.noise_model)

        return ImageRegionData.from_model_output(x, mmse.detach().cpu().numpy())

    def configure_optimizers(self) -> dict[str, Any]:  # type: ignore[override]
        """Configure optimizer and learning rate scheduler.

        Returns
        -------
        dict
            Optimizer and scheduler configuration.
        """
        return configure_optimizers(
            model=self.model,
            optimizer_name=self.config.optimizer.name,
            optimizer_parameters=self.config.optimizer.parameters,
            lr_scheduler_name=self.config.lr_scheduler.name,
            lr_scheduler_parameters=self.config.lr_scheduler.parameters,
            monitor=self.config.monitor_metric,
        )
