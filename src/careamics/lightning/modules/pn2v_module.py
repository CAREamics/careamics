"""Probabilistic Noise2Void Lightning module."""

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import lightning as L
import torch
from lightning.pytorch.trainer.states import TrainerFn
from torch import nn
from torchmetrics import MetricCollection

from careamics.config import PN2VAlgorithm
from careamics.config.factories.algorithm_factory import algorithm_factory
from careamics.dataset import ImageRegionData
from careamics.dataset.factory import TrainValData, TrainValSplitData
from careamics.dataset.normalization import (
    MeanStdNormalization,
    NoNormalization,
    RangeNormalization,
)
from careamics.dataset.normalization.normalization import Normalization
from careamics.dataset.normalization.utils import broadcast_stats
from careamics.lightning.logger import CoLogger
from careamics.losses import pn2v_loss
from careamics.losses.n2v_losses import pn2v_likelihoods
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
    log_images,
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
    noise_model: GaussianMixtureNoiseModel | MultiChannelNoiseModel,
) -> torch.Tensor:
    """Likelihood-weighted mean of the prior samples of each channel.

    The samples are channel-major: input channel `c` owns the samples
    `[c * K, (c + 1) * K)`, and they are weighted with the noise model of channel `c`.
    The result is the MMSE estimate used for the denoised image.

    Parameters
    ----------
    samples : torch.Tensor
        Denormalized prior samples, shape `(B, C * K, ...)`.
    observations : torch.Tensor
        Denormalized noisy observations, shape `(B, C, ...)`.
    noise_model : GaussianMixtureNoiseModel or MultiChannelNoiseModel
        Noise model with one channel per observation channel. A single
        `GaussianMixtureNoiseModel` is only valid for one channel.

    Returns
    -------
    torch.Tensor
        MMSE estimate, shape `(B, C, ...)`.
    """
    likelihoods = pn2v_likelihoods(samples, observations, noise_model)
    n_samples = samples.shape[1] // observations.shape[1]
    return torch.cat(
        [
            torch.sum(
                channel_likelihoods
                * samples[:, channel * n_samples : (channel + 1) * n_samples],
                dim=1,
                keepdim=True,
            )
            / torch.sum(channel_likelihoods, dim=1, keepdim=True)
            for channel, channel_likelihoods in enumerate(likelihoods)
        ],
        dim=1,
    )


class PN2VModule(L.LightningModule):
    """CAREamics PyTorch Lightning module for the PN2V algorithm.

    PN2V needs a trained noise model. It is not part of the configuration: attach it
    with `set_noise_model` (or `CAREamist.train(noise_model=...)`) before training or
    prediction. The noise model is saved in the checkpoints, outside the
    `state_dict`, and restored when a checkpoint is loaded.

    The UNet predicts `K` prior samples per pixel and input channel. The outputs are
    channel-major: input channel `c` owns the outputs `[c * K, (c + 1) * K)`, and
    is evaluated with the noise model of channel `c`. Predictions are the MMSE
    estimates, one per input channel.

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

        # Per-channel input statistics used to denormalize samples and observations,
        # cached at the start of each stage. They are plain attributes, not buffers,
        # so they are not saved in the `state_dict`.
        self._denorm_offset: torch.Tensor | None = None
        self._denorm_scale: torch.Tensor | None = None

        # Diagnostic metric on the prior mean of each channel. The PSNR of the MMSE
        # prediction is computed outside this module.
        n_channels = self.config.model.in_channels
        self.metrics: MetricCollection = MetricCollection(
            {
                f"SIPSNR_{i}": SIPSNR(
                    n_channels=n_channels,
                    output_channel=i,
                    use_scale_invariance=True,
                )
                for i in range(n_channels)
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

    def _noise_model(self) -> MultiChannelNoiseModel:
        """Return the noise model.

        This is a method and not a property, so that `hasattr` does not hide the
        error.

        Returns
        -------
        MultiChannelNoiseModel
            Noise model, one channel per input channel.

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
        return self._raw_noise_model

    def _cache_denormalization(self, normalization: Normalization) -> None:
        """Cache the per-channel input statistics of `normalization`.

        Samples and observations are always denormalized with the input statistics,
        also when the dataset has target statistics: the samples are predictions of
        the input. The arithmetic matches the `denormalize` method of each
        normalization.

        Parameters
        ----------
        normalization : Normalization
            Normalization of the dataset of the current stage.

        Raises
        ------
        TypeError
            If the normalization type is not supported.
        """
        n_channels = self.config.model.in_channels
        if isinstance(normalization, MeanStdNormalization):
            means = broadcast_stats(normalization.input_means, n_channels, "means")
            stds = broadcast_stats(normalization.input_stds, n_channels, "stds")
            offset = torch.tensor(means, dtype=torch.float32)
            scale = torch.tensor(stds, dtype=torch.float32) + normalization.eps
        elif isinstance(normalization, RangeNormalization):
            mins = broadcast_stats(normalization.input_mins, n_channels, "mins")
            maxes = broadcast_stats(normalization.input_maxes, n_channels, "maxes")
            offset = torch.tensor(mins, dtype=torch.float32)
            scale = torch.tensor(maxes, dtype=torch.float32) - offset
        elif isinstance(normalization, NoNormalization):
            offset = torch.zeros(n_channels, dtype=torch.float32)
            scale = torch.ones(n_channels, dtype=torch.float32)
        else:
            raise TypeError(
                f"PN2V does not support the normalization "
                f"{type(normalization).__name__}."
            )
        self._denorm_offset = offset.to(self.device)
        self._denorm_scale = scale.to(self.device)

    def _denormalize(self, array: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Denormalize a channel-major array with the cached input statistics.

        Parameters
        ----------
        array : torch.Tensor
            Normalized array, shape `(B, C * n_samples, ...)`.
        n_samples : int, default=1
            Number of consecutive channels per input channel.

        Returns
        -------
        torch.Tensor
            Denormalized array, same shape as `array`, in `float32`.

        Raises
        ------
        RuntimeError
            If the statistics have not been cached.
        """
        if self._denorm_offset is None or self._denorm_scale is None:
            raise RuntimeError(
                "The normalization statistics are not set. They are read from the "
                "datamodule at the start of fit, validation and prediction."
            )
        n_channels = self.config.model.in_channels
        batch, spatial = array.shape[0], array.shape[2:]
        stats_shape = (1, n_channels, 1) + (1,) * len(spatial)
        scale = self._denorm_scale.to(array.device).view(stats_shape)
        offset = self._denorm_offset.to(array.device).view(stats_shape)
        grouped = array.to(dtype=torch.float32).reshape(
            batch, n_channels, n_samples, *spatial
        )
        return (grouped * scale + offset).reshape(array.shape)

    def _prior_mean(self, prediction: torch.Tensor) -> torch.Tensor:
        """Mean of the scaled prior samples of each channel, in normalized space.

        Parameters
        ----------
        prediction : torch.Tensor
            Network output, shape `(B, C * K, ...)`.

        Returns
        -------
        torch.Tensor
            Prior mean, shape `(B, C, ...)`.
        """
        batch, spatial = prediction.shape[0], prediction.shape[2:]
        samples = (prediction * self.config.sample_scale).reshape(
            batch, self.config.model.in_channels, self.config.n_prior_samples, *spatial
        )
        return samples.mean(dim=2)

    def _denormalized_samples(
        self, prediction: torch.Tensor, original: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scale network outputs, then denormalize samples and original pixels.

        Parameters
        ----------
        prediction : torch.Tensor
            Network output in normalized space, shape `(B, C * K, ...)`.
        original : torch.Tensor
            Original noisy patch in normalized space, shape `(B, C, ...)`.

        Returns
        -------
        tuple of torch.Tensor
            Denormalized prior samples and denormalized observations.
        """
        samples = self._denormalize(
            prediction * self.config.sample_scale, self.config.n_prior_samples
        )
        observations = self._denormalize(original)
        return samples, observations

    def _start_stage(self, normalization: Normalization | None) -> None:
        """Check the noise model, move it to the device and cache the statistics.

        Parameters
        ----------
        normalization : Normalization or None
            Normalization of the dataset of the stage. If `None`, the cached
            statistics are kept.
        """
        self._noise_model().to_device(self.device)
        if normalization is not None:
            self._cache_denormalization(normalization)

    def on_fit_start(self) -> None:
        """Check the noise model and cache the training statistics.

        The training statistics are also used for validation during fit.
        """
        assert self._trainer is not None
        datamodule: CareamicsDataModule = self._trainer.datamodule  # type: ignore[union-attr]
        assert isinstance(datamodule._data, (TrainValData, TrainValSplitData))
        if datamodule._data.train_data_target is not None:
            logger.warning(
                "PN2V is a self-supervised algorithm, arguments passed to "
                "`train_data_target` will be ignored.",
            )
        self._start_stage(datamodule.train_dataset.normalization)  # type: ignore[union-attr]

    def on_validation_start(self) -> None:
        """Check the noise model, and cache the validation statistics.

        During fit, the training statistics cached in `on_fit_start` are kept.
        """
        normalization = None
        if self.trainer.state.fn == TrainerFn.VALIDATING:
            datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
            normalization = datamodule.val_dataset.normalization
        self._start_stage(normalization)

    def on_predict_start(self) -> None:
        """Check the noise model and cache the prediction statistics."""
        datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
        self._start_stage(datamodule.predict_dataset.normalization)

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
            Network output, `K` prior samples per input channel, channel-major.
        """
        return self.model(x)

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
        prior_mean = self._prior_mean(prediction)
        self.metrics(prior_mean, x_original)
        log_validation_stats(
            self, val_loss, batch_size=x_data.shape[0], metrics=self.metrics
        )

        not_sanity_checking = not self.trainer.sanity_checking
        if batch_idx == 0 and isinstance(self.logger, CoLogger) and not_sanity_checking:
            n_samples = min(5, x_data.shape[0])
            log_images(
                self,
                input_or_target=x_masked[:n_samples],
                prediction=prior_mean[:n_samples],
                is_supervised=False,
                denormalize=self._denormalize,
            )

    def predict_step(
        self,
        batch: tuple[ImageRegionData] | tuple[ImageRegionData, ImageRegionData],
        batch_idx: int,
    ) -> ImageRegionData:
        """Prediction step.

        The returned image is the MMSE estimate of each input channel, not the mean of
        the prior samples.

        Parameters
        ----------
        batch : ImageRegionData or (ImageRegionData, ImageRegionData)
            Input batch.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        ImageRegionData
            MMSE prediction in the original intensity range, one channel per input
            channel.
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
