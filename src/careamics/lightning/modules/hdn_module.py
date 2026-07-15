"""Hierarchical DivNoising (HDN) Lightning module."""

from typing import TYPE_CHECKING, Any, cast

import pytorch_lightning as L
import torch
from torch import nn
from torchmetrics import MetricCollection

from careamics.config import VAEBasedAlgorithm
from careamics.dataset import ImageRegionData
from careamics.dataset.normalization.mean_std_normalization import MeanStdNormalization
from careamics.losses.lvae import hdn_loss
from careamics.metrics import SIPSNR
from careamics.models.lvae.noise_models import (
    MultiChannelNoiseModel,
    multichannel_noise_model_factory,
)
from careamics.models.model_factory import model_factory
from careamics.utils.logging import get_logger

from .module_utils import configure_optimizers, log_training_stats, log_validation_stats

logger = get_logger(__name__)

if TYPE_CHECKING:
    from careamics.lightning.data.data_module import CareamicsDataModule


class HDNModule(L.LightningModule):
    """CAREamics PyTorch Lightning module for the HDN algorithm.

    HDN (Hierarchical DivNoising) is a VAE-based denoising algorithm built on the
    LVAE model. It is self-supervised by default (the input patch is used as its own
    target), but a supervised target can be provided through the second batch element.

    The reconstruction likelihood is selected automatically from the configuration:

    - if a noise model is provided, the noise model likelihood is used
      (`predict_logvar` must be `False`);
    - otherwise a Gaussian likelihood with a learned per-pixel variance is used
      (`predict_logvar` must be `True`).

    Parameters
    ----------
    algorithm_config : VAEBasedAlgorithm or dict
        Configuration for the HDN algorithm, either as a `VAEBasedAlgorithm` instance
        or a dictionary.
    """

    def __init__(self, algorithm_config: VAEBasedAlgorithm | dict[str, Any]) -> None:
        """Instantiate HDNModule.

        Parameters
        ----------
        algorithm_config : VAEBasedAlgorithm or dict
            Configuration for the HDN algorithm, either as a `VAEBasedAlgorithm`
            instance or a dictionary.
        """
        super().__init__()

        if isinstance(algorithm_config, dict):
            config = VAEBasedAlgorithm(**algorithm_config)
        else:
            config = algorithm_config

        if not isinstance(config, VAEBasedAlgorithm):
            raise TypeError("algorithm_config must be a VAEBasedAlgorithm")

        self.save_hyperparameters({"algorithm_config": config.model_dump(mode="json")})
        self.config: VAEBasedAlgorithm = config

        self.model: nn.Module = model_factory(self.config.model)
        self.loss_func = hdn_loss

        self.noise_model: MultiChannelNoiseModel | None = (
            multichannel_noise_model_factory(self.config.noise_model)
        )

        self.predict_logvar: bool = self.config.model.predict_logvar
        if self.noise_model is None and not self.predict_logvar:
            raise ValueError(
                "Without a noise model, HDN learns a Gaussian likelihood and "
                "requires `predict_logvar=True`."
            )
        if self.noise_model is not None and self.predict_logvar:
            raise ValueError(
                "With a noise model, HDN uses the noise model likelihood and "
                "requires `predict_logvar=False`."
            )

        self.data_mean: float | None = None
        self.data_std: float | None = None

        self.metrics: MetricCollection = MetricCollection(
            {
                f"SIPSNR_{i}": SIPSNR(
                    n_channels=self.config.model.output_channels,
                    output_channel=i,
                    use_scale_invariance=True,
                )
                for i in range(self.config.model.output_channels)
            }
        )

    def on_fit_start(self) -> None:
        """On fit start hook for HDN module.

        Raises
        ------
        TypeError
            If a noise model is used with a normalization other than
            `MeanStdNormalization`.
        """
        if self.noise_model is None:
            return
        assert self._trainer is not None
        datamodule: CareamicsDataModule = self._trainer.datamodule  # type: ignore[union-attr]
        # The noise model likelihood is parameterized in mean/std space, so the noise
        # model path requires zero-mean/unit-variance normalization to recover the
        # data statistics; other normalizations do not expose input means/stds.
        normalization = datamodule.train_dataset.normalization  # type: ignore[union-attr]
        if not isinstance(normalization, MeanStdNormalization):
            raise TypeError(
                "HDN with a noise model requires MeanStdNormalization to recover the "
                f"data statistics, but got {type(normalization).__name__}."
            )
        self.data_mean = float(normalization.input_means[0])
        self.data_std = float(normalization.input_stds[0])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        tuple of (torch.Tensor, dict)
            The reconstruction and the top-down layer data.
        """
        return self.model(x)

    def _get_target(
        self,
        batch: tuple[ImageRegionData] | tuple[ImageRegionData, ImageRegionData],
        x_data: torch.Tensor,
    ) -> torch.Tensor:
        """Return the training target.

        In the self-supervised case the input itself is the target; in the supervised
        case the target is taken from the second batch element.

        Parameters
        ----------
        batch : ImageRegionData or (ImageRegionData, ImageRegionData)
            The input batch.
        x_data : torch.Tensor
            The input tensor (used as target in the self-supervised case).

        Returns
        -------
        torch.Tensor
            Target tensor.
        """
        if not self.config.is_supervised:
            return x_data
        supervised_batch = cast("tuple[ImageRegionData, ImageRegionData]", batch)
        target_data = supervised_batch[1].data
        # batch data has been collated into a tensor by the DataLoader
        assert isinstance(target_data, torch.Tensor)
        return target_data

    def _compute_loss(
        self, model_outputs: tuple[torch.Tensor, dict[str, Any]], target: torch.Tensor
    ) -> dict[str, torch.Tensor] | None:
        """Compute the HDN loss dictionary.

        Parameters
        ----------
        model_outputs : tuple of (torch.Tensor, dict)
            The model reconstruction and top-down data.
        target : torch.Tensor
            Target tensor.

        Returns
        -------
        dict of str to torch.Tensor or None
            Dictionary with `loss`, `reconstruction_loss` and `kl_loss`, or
            `None` if the loss is NaN (so the caller can skip the batch).

        Raises
        ------
        RuntimeError
            If the noise model is used but its data statistics are not set.
        """
        if self.noise_model is not None and (
            self.data_mean is None or self.data_std is None
        ):
            raise RuntimeError(
                "Data statistics are missing; they are set in `on_fit_start`. "
                "Call `trainer.fit` before computing the loss."
            )
        return self.loss_func(
            model_outputs=model_outputs,
            targets=target,
            config=self.config.loss,
            noise_model=self.noise_model,
            data_mean=self.data_mean,
            data_std=self.data_std,
        )

    def _get_reconstruction(
        self, model_outputs: tuple[torch.Tensor, dict[str, Any]]
    ) -> torch.Tensor:
        """Extract the reconstructed mean from the model outputs.

        If noise model is not used, the model predicts the log-variance, so the output
        channels are split into mean and log-variance; only the mean is returned.

        Parameters
        ----------
        model_outputs : tuple of (torch.Tensor, dict)
            The model reconstruction and top-down data.

        Returns
        -------
        torch.Tensor
            Reconstructed mean, shape (B, output_channels, [Z], Y, X).
        """
        predictions, _ = model_outputs
        if self.predict_logvar:
            predictions = predictions.chunk(2, dim=1)[0]
        return predictions

    def training_step(
        self,
        batch: tuple[ImageRegionData] | tuple[ImageRegionData, ImageRegionData],
        batch_idx: int,
    ) -> torch.Tensor | None:
        """Training step for HDN.

        Parameters
        ----------
        batch : ImageRegionData or (ImageRegionData, ImageRegionData)
            A tuple containing the input data and, optionally, the target data.
        batch_idx : int
            The index of the current batch in the training loop.

        Returns
        -------
        torch.Tensor or None
            The loss value for the current training step, or `None` to skip the
            batch when the loss is NaN.
        """
        x_data = batch[0].data
        # batch data has been collated into a tensor by the DataLoader
        assert isinstance(x_data, torch.Tensor)
        target = self._get_target(batch, x_data)

        model_outputs = self.model(x_data)
        loss = self._compute_loss(model_outputs, target)
        if loss is None:  # skip the batch on NaN loss
            return None

        log_training_stats(self, loss["loss"], batch_size=x_data.shape[0])
        self.log_dict(
            {f"train_{k}": v for k, v in loss.items() if k != "loss"},
            on_step=True,
            on_epoch=True,
            batch_size=x_data.shape[0],
        )
        return loss["loss"]

    def validation_step(
        self,
        batch: tuple[ImageRegionData] | tuple[ImageRegionData, ImageRegionData],
        batch_idx: int,
    ) -> None:
        """Validation step for HDN.

        Parameters
        ----------
        batch : ImageRegionData or (ImageRegionData, ImageRegionData)
            A tuple containing the input data and, optionally, the target data.
        batch_idx : int
            The index of the current batch in the validation loop.
        """
        x_data = batch[0].data
        # batch data has been collated into a tensor by the DataLoader
        assert isinstance(x_data, torch.Tensor)
        target = self._get_target(batch, x_data)

        model_outputs = self.model(x_data)
        val_loss = self._compute_loss(model_outputs, target)
        if val_loss is None:  # skip the batch on NaN loss
            return

        self.metrics(self._get_reconstruction(model_outputs), target)
        log_validation_stats(
            self, val_loss["loss"], batch_size=x_data.shape[0], metrics=self.metrics
        )

    def predict_step(
        self,
        batch: tuple[ImageRegionData] | tuple[ImageRegionData, ImageRegionData],
        batch_idx: int,
    ) -> ImageRegionData:
        """Prediction step for HDN.

        Parameters
        ----------
        batch : ImageRegionData or (ImageRegionData, ImageRegionData)
            A tuple containing the input data and, optionally, the target data.
        batch_idx : int
            The index of the current batch in the prediction loop.

        Returns
        -------
        ImageRegionData
            The output batch containing the reconstruction.
        """
        x = batch[0]
        x_data = x.data
        # batch data has been collated into a tensor by the DataLoader
        assert isinstance(x_data, torch.Tensor)

        # reconfigure the model for the current input spatial size
        self.model.reset_for_inference(x_data.shape[-2:])

        prediction = self._get_reconstruction(self.model(x_data))

        normalization = self._trainer.datamodule.predict_dataset.normalization  # type: ignore[union-attr]
        denormalized_output = (
            normalization.denormalize(prediction).detach().cpu().numpy()
        )

        return ImageRegionData(
            data=denormalized_output,
            source=x.source,
            data_shape=x.data_shape,
            dtype=x.dtype,
            axes=x.axes,
            region_spec=x.region_spec,
            additional_metadata=x.additional_metadata,
            original_data_shape=x.original_data_shape,
        )

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
            monitor="val_loss",
        )
