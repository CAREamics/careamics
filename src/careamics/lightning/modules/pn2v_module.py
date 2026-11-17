"""Probabilistic Noise2Void Lightning module."""

from collections.abc import Sequence
from pathlib import Path
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
from careamics.models.lvae.noise_models import (
    GaussianMixtureNoiseModel,
    MultiChannelNoiseModel,
)
from careamics.models.unet import UNet
from careamics.utils.logging import get_logger

from .module_utils import (
    configure_optimizers,
    load_noise_model_from_checkpoint,
    log_training_stats,
    log_validation_stats,
    resolve_noise_model,
    save_noise_model_to_checkpoint,
)
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

    PN2V needs a trained noise model. It is not part of the configuration: attach it
    with `set_noise_model` (or `CAREamist.train(noise_model=...)`) before training or
    prediction. The noise model is saved in the checkpoints, outside the
    `state_dict`, and restored when a checkpoint is loaded.

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
        self.loss_func = pn2v_loss

        # Raw-space noise model. Set with `object.__setattr__` so that it is not
        # registered as a submodule and stays out of the `state_dict`.
        self._raw_noise_model: MultiChannelNoiseModel | None = None

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

    def set_noise_model(
        self, noise_model: MultiChannelNoiseModel | Sequence[str | Path]
    ) -> None:
        """Attach the trained noise model used by the loss and the MMSE estimate.

        Parameters
        ----------
        noise_model : MultiChannelNoiseModel or sequence of str or Path
            The trained (raw-space) noise model, or the per-channel `.npz` paths to
            load it from. It needs one channel per input channel.

        Raises
        ------
        ValueError
            If the number of noise model channels is not the number of input
            channels.
        """
        nm = resolve_noise_model(noise_model)
        n_channels = self.config.model.in_channels
        if len(nm) != n_channels:
            raise ValueError(
                f"PN2V needs one noise model channel per input channel. The noise "
                f"model has {len(nm)} channel(s), the model has {n_channels} input "
                f"channel(s)."
            )
        # avoid nn.Module registration and keep out of `state_dict`
        object.__setattr__(self, "_raw_noise_model", nm)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Save the raw-space noise model in the checkpoint.

        Parameters
        ----------
        checkpoint : dict
            The checkpoint dictionary being saved.
        """
        save_noise_model_to_checkpoint(self._raw_noise_model, checkpoint)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Restore the raw-space noise model from the checkpoint, if present.

        Parameters
        ----------
        checkpoint : dict
            The checkpoint dictionary being loaded.
        """
        restored = load_noise_model_from_checkpoint(checkpoint)
        if restored is not None:
            object.__setattr__(self, "_raw_noise_model", restored)

    def _noise_model(self) -> GaussianMixtureNoiseModel:
        """Return the noise model of the input channel.

        This is a method and not a property, so that `hasattr` does not hide the
        error.

        Returns
        -------
        GaussianMixtureNoiseModel
            Noise model of the input channel.

        Raises
        ------
        ValueError
            If no noise model is set.
        """
        if self._raw_noise_model is None:
            raise ValueError(
                "PN2V requires a noise model. Provide one with "
                "`CAREamist.train(noise_model=...)` or `PN2VModule.set_noise_model`."
            )
        return cast(GaussianMixtureNoiseModel, self._raw_noise_model.nmodel_0)

    def _noise_model_to_device(self) -> None:
        """Check that a noise model is set and move it to the module device."""
        self._noise_model()
        assert self._raw_noise_model is not None
        self._raw_noise_model.to_device(self.device)

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
        self._noise_model_to_device()

    def on_validation_start(self) -> None:
        """On validation start hook for the PN2V module."""
        self._noise_model_to_device()

    def on_predict_start(self) -> None:
        """On predict start hook for the PN2V module."""
        self._noise_model_to_device()

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
        loss = self.loss_func(samples, observations, mask, self._noise_model())

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
        val_loss = self.loss_func(samples, observations, mask, self._noise_model())
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
        mmse = pn2v_mmse(samples, observations, self._noise_model())

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
