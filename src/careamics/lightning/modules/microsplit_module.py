"""MicroSplit Lightning module."""

from typing import TYPE_CHECKING, Any, cast

import pytorch_lightning as L
import torch
from torch import nn
from torchmetrics import MetricCollection

from careamics.config import MicroSplitAlgorithm
from careamics.dataset import ImageRegionData
from careamics.dataset.factory import TrainValData, TrainValSplitData
from careamics.dataset.normalization.mean_std_normalization import MeanStdNormalization
from careamics.losses.lvae import microsplit_loss
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


class MicroSplitModule(L.LightningModule):
    """CAREamics PyTorch Lightning module for the MicroSplit algorithm.

    MicroSplit is a supervised VAE-based algorithm that unmixes a single input
    channel into several target channels, built on the LVAE model. The reconstruction
    likelihood is the weighted combination configured by the loss:

    - `musplit_weight` weights a Gaussian likelihood with a learned per-pixel
      variance (requires `predict_logvar=True`);
    - `denoisplit_weight` weights a noise model likelihood (requires a noise model
      and `MeanStdNormalization`; the noise model is transformed into normalized
      data space at the start of training).

    Parameters
    ----------
    algorithm_config : MicroSplitAlgorithm or dict
        Configuration for the MicroSplit algorithm, either as a `MicroSplitAlgorithm`
        instance or a dictionary.
    """

    def __init__(self, algorithm_config: MicroSplitAlgorithm | dict[str, Any]) -> None:
        """Instantiate MicroSplitModule.

        Parameters
        ----------
        algorithm_config : MicroSplitAlgorithm or dict
            Configuration for the MicroSplit algorithm, either as a
            `MicroSplitAlgorithm` instance or a dictionary.
        """
        super().__init__()

        if isinstance(algorithm_config, dict):
            config = MicroSplitAlgorithm(**algorithm_config)
        else:
            config = algorithm_config

        if not isinstance(config, MicroSplitAlgorithm):
            raise TypeError("algorithm_config must be a MicroSplitAlgorithm")

        self.save_hyperparameters({"algorithm_config": config.model_dump(mode="json")})
        self.config = config

        self.model: nn.Module = model_factory(self.config.model)
        self.loss_func = microsplit_loss

        self.noise_model: MultiChannelNoiseModel | None = (
            multichannel_noise_model_factory(self.config.noise_model)
        )
        self.predict_logvar: bool = self.config.model.predict_logvar

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
        """Validate the supervised-training and noise model requirements.

        When the noise model likelihood is used, the noise model is rebuilt from the
        (raw-space) configuration and transformed into normalized data space using
        each channel's statistics from the training dataset normalization, which must
        therefore be `MeanStdNormalization`. Rebuilding from the configuration keeps
        this hook idempotent (e.g. when resuming from a checkpoint).

        Raises
        ------
        ValueError
            If target data is missing (MicroSplit is supervised), or if the noise
            model likelihood is used without a noise model.
        TypeError
            If the noise model likelihood is used with a normalization other than
            `MeanStdNormalization`.
        """
        assert self._trainer is not None
        datamodule: CareamicsDataModule = self._trainer.datamodule  # type: ignore[union-attr]
        assert isinstance(datamodule._data, (TrainValData, TrainValSplitData))
        if datamodule._data.train_data_target is None:
            raise ValueError(
                "MicroSplit is supervised: `train_data_target` must be provided."
            )
        if (
            isinstance(datamodule._data, TrainValData)
            and datamodule._data.val_data_target is None
        ):
            raise ValueError(
                "MicroSplit is supervised: `val_data_target` must be provided."
            )
        if self.config.loss.denoisplit_weight > 0:
            if self.noise_model is None:
                raise ValueError(
                    "The noise model likelihood (denoisplit_weight > 0) requires a "
                    "noise model. Provide one in the configuration."
                )
            # the noise model likelihood operates in normalized data space, so the
            # per-channel statistics are recovered from the training normalization
            normalization = datamodule.train_dataset.normalization  # type: ignore[union-attr]
            if not isinstance(normalization, MeanStdNormalization):
                raise TypeError(
                    "MicroSplit with a noise model requires MeanStdNormalization to "
                    f"recover the data statistics, but got "
                    f"{type(normalization).__name__}."
                )
            # target statistics when supervised, input statistics otherwise
            means = normalization.target_means or normalization.input_means
            stds = normalization.target_stds or normalization.input_stds
            # rebuild the raw-space model from the config, then normalize each
            # channel's model with that channel's statistics
            raw_noise_model = multichannel_noise_model_factory(self.config.noise_model)
            assert raw_noise_model is not None
            self.noise_model = raw_noise_model.get_normalized_copy(
                [float(m) for m in means], [float(s) for s in stds]
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

    def _compute_loss(
        self, model_outputs: tuple[torch.Tensor, dict[str, Any]], target: torch.Tensor
    ) -> dict[str, torch.Tensor] | None:
        """Compute the MicroSplit loss dictionary.

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
        """
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

        When the model predicts the log-variance, the output channels are split into
        mean and log-variance; only the mean is returned.

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
        self, batch: tuple[ImageRegionData, ImageRegionData], batch_idx: int
    ) -> torch.Tensor | None:
        """Training step for MicroSplit.

        Parameters
        ----------
        batch : (ImageRegionData, ImageRegionData)
            A tuple containing the input data and the target data.
        batch_idx : int
            The index of the current batch in the training loop.

        Returns
        -------
        torch.Tensor or None
            The loss value, or `None` to skip the batch when the loss is NaN.
        """
        x_data = batch[0].data
        target = batch[1].data
        # batch data has been collated into tensors by the DataLoader
        assert isinstance(x_data, torch.Tensor)
        assert isinstance(target, torch.Tensor)

        model_outputs = self.model(x_data)
        loss = self._compute_loss(model_outputs, target)
        if loss is None:
            # skip the batch on NaN loss
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
        self, batch: tuple[ImageRegionData, ImageRegionData], batch_idx: int
    ) -> None:
        """Validation step for MicroSplit.

        Parameters
        ----------
        batch : (ImageRegionData, ImageRegionData)
            A tuple containing the input data and the target data.
        batch_idx : int
            The index of the current batch in the validation loop.
        """
        x_data = batch[0].data
        target = batch[1].data
        # batch data has been collated into tensors by the DataLoader
        assert isinstance(x_data, torch.Tensor)
        assert isinstance(target, torch.Tensor)

        model_outputs = self.model(x_data)
        val_loss = self._compute_loss(model_outputs, target)
        if val_loss is None:
            # skip the batch on NaN loss
            return

        self.metrics(self._get_reconstruction(model_outputs), target)
        log_validation_stats(
            self, val_loss["loss"], batch_size=x_data.shape[0], metrics=self.metrics
        )

    def predict_step(
        self, batch: tuple[ImageRegionData, ...], batch_idx: int
    ) -> ImageRegionData:
        """Prediction step for MicroSplit.

        Runs a single forward pass and returns the reconstruction denormalized into
        target space.

        Parameters
        ----------
        batch : tuple of ImageRegionData
            The input batch; only the first element (input region) is used.
        batch_idx : int
            The index of the current batch in the prediction loop.

        Returns
        -------
        ImageRegionData
            The output batch containing the reconstruction, with the channel
            dimension of `data_shape` set to the number of output channels.
        """
        x = batch[0]
        x_data = x.data
        # batch data has been collated into a tensor by the DataLoader
        assert isinstance(x_data, torch.Tensor)

        # reconfigure the model for the current input spatial size
        n_spatial_dims = x_data.dim() - 2
        self.model.reset_for_inference(tuple(x_data.shape[-n_spatial_dims:]))

        prediction = self._get_reconstruction(self.model(x_data))

        # denormalize into target space using the prediction dataset's normalization
        # (uses target statistics when available), consistent with the other modules
        normalization = self._trainer.datamodule.predict_dataset.normalization  # type: ignore[union-attr]
        denormalized_output = (
            normalization.denormalize(prediction).detach().cpu().numpy()
        )

        output_channels = self.config.model.output_channels
        output_data_shape = list(x.data_shape)
        output_data_shape[1] = torch.full_like(
            cast(torch.Tensor, output_data_shape[1]), output_channels
        )

        return ImageRegionData(
            data=denormalized_output,
            source=x.source,
            data_shape=output_data_shape,
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
