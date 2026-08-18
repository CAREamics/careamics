"""Hierarchical DivNoising (HDN) Lightning module."""

from functools import partial
from typing import TYPE_CHECKING, Any, cast

import lightning.pytorch as L
import torch
from torchmetrics import MetricCollection

from careamics.config import HDNAlgorithm
from careamics.dataset import ImageRegionData
from careamics.dataset.normalization.mean_std_normalization import MeanStdNormalization
from careamics.dataset.normalization.normalization import Normalization
from careamics.losses.lvae import hdn_loss
from careamics.metrics import SIPSNR
from careamics.models.lvae import LadderVAE
from careamics.models.lvae.noise_models import (
    MultiChannelNoiseModel,
    multichannel_noise_model_factory,
)
from careamics.models.model_factory import model_factory
from careamics.utils.logging import get_logger

from .module_utils import (
    configure_optimizers,
    log_training_stats,
    log_validation_stats,
    mmse_and_sample_std,
)

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

    HDN is stochastic: every forward pass draws a fresh latent sample. Set the
    `n_samples` attribute to average several draws at prediction time and obtain
    their standard deviation as an uncertainty estimate.

    Parameters
    ----------
    algorithm_config : HDNAlgorithm or dict
        Configuration for the HDN algorithm, either as an `HDNAlgorithm` instance
        or a dictionary.
    """

    def __init__(self, algorithm_config: HDNAlgorithm | dict[str, Any]) -> None:
        """Instantiate HDNModule.

        Parameters
        ----------
        algorithm_config : HDNAlgorithm or dict
            Configuration for the HDN algorithm, either as an `HDNAlgorithm`
            instance or a dictionary.
        """
        super().__init__()

        if isinstance(algorithm_config, dict):
            config = HDNAlgorithm(**algorithm_config)
        else:
            config = algorithm_config

        if not isinstance(config, HDNAlgorithm):
            raise TypeError("algorithm_config must be an HDNAlgorithm")

        self.save_hyperparameters({"algorithm_config": config.model_dump(mode="json")})
        self.config: HDNAlgorithm = config

        self.model: LadderVAE = cast("LadderVAE", model_factory(self.config.model))
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

        self.n_samples: int = 1

    def on_fit_start(self) -> None:
        """On fit start hook for HDN module.

        When a noise model is used, it is rebuilt from the (raw-space) configuration
        and transformed into normalized data space using the input statistics from
        the training dataset normalization. Rebuilding from the configuration keeps
        this hook idempotent (e.g. when resuming from a checkpoint).

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
        # The noise model likelihood operates in normalized data space, so the noise
        # model path requires zero-mean/unit-variance normalization to recover the
        # data statistics; other normalizations do not expose input means/stds.
        normalization = datamodule.train_dataset.normalization  # type: ignore[union-attr]
        if not isinstance(normalization, MeanStdNormalization):
            raise TypeError(
                "HDN with a noise model requires MeanStdNormalization to recover the "
                f"data statistics, but got {type(normalization).__name__}."
            )
        raw_noise_model = multichannel_noise_model_factory(self.config.noise_model)
        assert raw_noise_model is not None
        self.noise_model = raw_noise_model.get_normalized_copy(
            [float(normalization.input_means[0])],
            [float(normalization.input_stds[0])],
        )

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
        if not self.config.is_supervised():
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
            If the noise model is used but has not been normalized yet.
        """
        if self.noise_model is not None and not self.noise_model.is_normalized:
            raise RuntimeError(
                "The noise model has not been normalized into the data space; "
                "this happens in `on_fit_start`. Call `trainer.fit` before "
                "computing the loss."
            )
        return self.loss_func(
            model_outputs=model_outputs,
            targets=target,
            config=self.config.loss,
            noise_model=self.noise_model,
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

    def predict_sample(
        self, x_data: torch.Tensor, normalization: Normalization
    ) -> torch.Tensor:
        """Draw one reconstruction and denormalize it into target space.

        Each call draws a new latent sample, so repeated calls give different
        reconstructions.

        Parameters
        ----------
        x_data : torch.Tensor
            Input tensor, shape (B, C, [Z], Y, X).
        normalization : Normalization
            Normalization used to map the reconstruction back into target space.

        Returns
        -------
        torch.Tensor
            A single reconstruction, shape (B, output_channels, [Z], Y, X).
        """
        reconstruction = self._get_reconstruction(self.model(x_data))
        return normalization.denormalize(reconstruction)

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
    ) -> tuple[ImageRegionData, ImageRegionData | None]:
        """Prediction step for HDN.

        Parameters
        ----------
        batch : ImageRegionData or (ImageRegionData, ImageRegionData)
            A tuple containing the input data and, optionally, the target data.
        batch_idx : int
            The index of the current batch in the prediction loop.

        Returns
        -------
        tuple of (ImageRegionData, ImageRegionData or None)
            The output batch containing the reconstruction and the
            uncertainty estimate if several samples were drawn.
        """
        x = batch[0]
        x_data = x.data
        # batch data has been collated into a tensor by the DataLoader
        assert isinstance(x_data, torch.Tensor)

        # reconfigure the model for the current input spatial size
        n_spatial_dims = x_data.dim() - 2
        self.model.reset_for_inference(tuple(x_data.shape[-n_spatial_dims:]))

        normalization = self._trainer.datamodule.predict_dataset.normalization  # type: ignore[union-attr]
        mean, std = mmse_and_sample_std(
            partial(self.predict_sample, normalization=normalization),
            x_data,
            self.n_samples,
        )

        output_channels = self.config.model.output_channels
        output_data_shape: list[Any] = list(x.data_shape)
        output_data_shape[1] = torch.full_like(
            cast(torch.Tensor, output_data_shape[1]), output_channels
        )

        prediction = ImageRegionData(
            data=mean.cpu().numpy(),
            source=x.source,
            data_shape=output_data_shape,
            dtype=x.dtype,
            axes=x.axes,
            target_axes=x.target_axes,
            region_spec=x.region_spec,
            additional_metadata=x.additional_metadata,
            original_data_shape=x.original_data_shape,
        )
        if std is None:
            return prediction, None
        uncertainty = prediction._replace(data=std.cpu().numpy())
        return prediction, uncertainty

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
