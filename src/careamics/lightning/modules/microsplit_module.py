"""MicroSplit Lightning module."""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import pytorch_lightning as L
import torch

from careamics.compat.transforms.normalize import Denormalize
from careamics.config import (
    VAEBasedAlgorithm,
)
from careamics.dataset import ImageRegionData
from careamics.dataset.factory import TrainValData, TrainValSplitData
from careamics.lightning.modules.module_utils import (
    get_optimizer,
    get_scheduler,
)
from careamics.losses.lvae import lvae_loss_factory
from careamics.metrics.metrics import RunningPSNR, scale_invariant_psnr
from careamics.models.lvae.likelihoods import (
    GaussianLikelihood,
    NoiseModelLikelihood,
    likelihood_factory,
)
from careamics.models.lvae.noise_models import (
    GaussianMixtureNoiseModel,
    MultiChannelNoiseModel,
    multichannel_noise_model_factory,
)
from careamics.models.model_factory import model_factory

if TYPE_CHECKING:
    from careamics.lightning.data.data_module import CareamicsDataModule

NoiseModel = Union[GaussianMixtureNoiseModel, MultiChannelNoiseModel]

# TODO Denormalize is still imported from careamics.compat; replace once
# Normalization.denormalize supports target-space stats.


class MicroSplitModule(L.LightningModule):
    """
    MicroSplit Lightning module.

    This class encapsulates the a PyTorch model along with the training, validation,
    and testing logic. It is configured using an `AlgorithmModel` Pydantic class.

    Parameters
    ----------
    algorithm_config : Union[VAEAlgorithmConfig, dict]
        Algorithm configuration.

    Attributes
    ----------
    model : nn.Module
        PyTorch model.
    loss_func : nn.Module
        Loss function.
    optimizer_name : str
        Optimizer name.
    optimizer_params : dict
        Optimizer parameters.
    lr_scheduler_name : str
        Learning rate scheduler name.
    """

    def __init__(self, algorithm_config: Union[VAEBasedAlgorithm, dict]) -> None:
        """Lightning module for CAREamics.

        This class encapsulates the a PyTorch model along with the training, validation,
        and testing logic. It is configured using an `AlgorithmModel` Pydantic class.

        Parameters
        ----------
        algorithm_config : Union[AlgorithmModel, dict]
            Algorithm configuration.
        """
        super().__init__()
        # if loading from a checkpoint, AlgorithmModel needs to be instantiated
        self.algorithm_config = (
            VAEBasedAlgorithm(**algorithm_config)
            if isinstance(algorithm_config, dict)
            else algorithm_config
        )

        # create model
        self.model: torch.nn.Module = model_factory(self.algorithm_config.model)

        # create noise model (VAE algorithms always use multichannel nm factory)
        self.noise_model: NoiseModel | None = multichannel_noise_model_factory(
            self.algorithm_config.noise_model
        )

        self.noise_model_likelihood: NoiseModelLikelihood | None = None
        if self.algorithm_config.noise_model_likelihood is not None:
            self.noise_model_likelihood = likelihood_factory(
                config=self.algorithm_config.noise_model_likelihood,
                noise_model=self.noise_model,
            )

        self.gaussian_likelihood: GaussianLikelihood | None = likelihood_factory(
            self.algorithm_config.gaussian_likelihood
        )

        self.loss_parameters = self.algorithm_config.loss
        self.loss_func = lvae_loss_factory(self.algorithm_config.loss.loss_type)

        # save optimizer and lr_scheduler names and parameters
        self.optimizer_name = self.algorithm_config.optimizer.name
        self.optimizer_params = self.algorithm_config.optimizer.parameters
        self.lr_scheduler_name = self.algorithm_config.lr_scheduler.name
        self.lr_scheduler_params = self.algorithm_config.lr_scheduler.parameters

        # initialize running PSNR
        self.running_psnr = [
            RunningPSNR() for _ in range(self.algorithm_config.model.output_channels)
        ]

        # target-channel denormalization stats used by `predict_step`. Set via
        # `set_target_stats(...)` for standalone prediction, or auto-populated in
        # `on_predict_start` from the trainer's data module when running through
        # `Trainer.predict(...)`.
        self.target_means: list[float] | None = None
        self.target_stds: list[float] | None = None

    def on_fit_start(self) -> None:
        """On fit start hook.

        Check that training and validation target data have been supplied, since
        MicroSplit is trained in a fully supervised manner.
        """
        assert self._trainer is not None
        datamodule: CareamicsDataModule = self._trainer.datamodule  # type: ignore[union-attr]
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, (1 + n_LC), [Z], Y, X), where n_LC is the
            number of lateral inputs.

        Returns
        -------
        tuple[torch.Tensor, dict[str, Any]]
            A tuple with the output tensor and additional data from the top-down pass.
        """
        return self.model(x)

    def set_data_stats(self, data_mean, data_std):
        """Set data mean and std for the noise model likelihood.

        Parameters
        ----------
        data_mean : float
            Mean of the data.
        data_std : float
            Standard deviation of the data.
        """
        if self.noise_model_likelihood is not None:
            self.noise_model_likelihood.set_data_stats(data_mean, data_std)

    def set_target_stats(
        self,
        target_means: Sequence[float],
        target_stds: Sequence[float],
    ) -> None:
        """Set per-target-channel mean / std used to denormalize predictions.

        Required for standalone calls to `predict_step` outside a Lightning Trainer.
        When running through `Trainer.predict(...)`, `on_predict_start` will
        auto-populate these from the data module's prediction-dataset normalization
        if they are still unset.

        Parameters
        ----------
        target_means : Sequence[float]
            Per-target-channel means in target space.
        target_stds : Sequence[float]
            Per-target-channel standard deviations in target space.
        """
        self.target_means = np.atleast_1d(np.asarray(target_means)).tolist()
        self.target_stds = np.atleast_1d(np.asarray(target_stds)).tolist()

    def on_predict_start(self) -> None:
        """Auto-populate target denormalization stats from the trainer if unset.

        Preserves the existing `Trainer.predict(...)` flow (where stats live on the
        data module's prediction-dataset normalization) for callers that don't
        invoke `set_target_stats` explicitly.
        """
        if self.target_means is not None and self.target_stds is not None:
            return
        trainer = getattr(self, "_trainer", None)
        if trainer is None:
            return
        datamodule = getattr(trainer, "datamodule", None)
        predict_dataset = getattr(datamodule, "predict_dataset", None)
        normalization = getattr(predict_dataset, "normalization", None)
        target_means = getattr(normalization, "target_means", None)
        target_stds = getattr(normalization, "target_stds", None)
        if target_means is not None and target_stds is not None:
            self.set_target_stats(target_means, target_stds)

    def training_step(
        self, batch: tuple[ImageRegionData, ImageRegionData], batch_idx: Any
    ) -> dict[str, torch.Tensor] | None:
        """Training step.

        Parameters
        ----------
        batch : tuple[ImageRegionData, ImageRegionData]
            Input batch. It is a tuple with the input data and the target data.
            The input data has shape (B, (1 + n_LC), [Z], Y, X), where n_LC is the
            number of lateral inputs. The target data has shape (B, C, [Z], Y, X),
            where C is the number of target channels (e.g., 1 in HDN, >1 in
            muSplit/denoiSplit).
        batch_idx : Any
            Batch index.

        Returns
        -------
        Any
            Loss value.
        """
        x, target = batch[0], batch[1]

        # Forward pass
        out = self.model(x.data)
        target = target.data

        # Update loss parameters
        self.loss_parameters.kl_params.current_epoch = self.current_epoch

        # Compute loss
        if self.noise_model_likelihood is not None:
            if (
                self.noise_model_likelihood.data_mean is None
                or self.noise_model_likelihood.data_std is None
            ):
                raise RuntimeError(
                    "NoiseModelLikelihood: mean and std must be set before training."
                )
        loss = self.loss_func(
            model_outputs=out,
            targets=target,
            config=self.loss_parameters,
            gaussian_likelihood=self.gaussian_likelihood,
            noise_model_likelihood=self.noise_model_likelihood,
        )

        # Logging
        # TODO: implement a separate logging method?
        self.log_dict(loss, on_step=True, on_epoch=True)

        try:
            optimizer = self.optimizers()
            current_lr = optimizer.param_groups[0]["lr"]
            self.log(
                "learning_rate", current_lr, on_step=False, on_epoch=True, logger=True
            )
        except RuntimeError:
            # This happens when the module is not attached to a trainer, e.g., in tests
            pass
        return loss

    def validation_step(
        self, batch: tuple[ImageRegionData, ImageRegionData], batch_idx: Any
    ) -> None:
        """Validation step.

        Parameters
        ----------
        batch : tuple[ImageRegionData, ImageRegionData]
            Input batch. It is a tuple with the input data and the target data.
            The input data has shape (B, (1 + n_LC), [Z], Y, X), where n_LC is the
            number of lateral inputs. The target data has shape (B, C, [Z], Y, X),
            where C is the number of target channels (e.g., 1 in HDN, >1 in
            muSplit/denoiSplit).
        batch_idx : Any
            Batch index.
        """
        x, target = batch[0], batch[1]

        # Forward pass
        out = self.model(x.data)
        target = target.data

        # Compute loss
        loss = self.loss_func(
            model_outputs=out,
            targets=target,
            config=self.loss_parameters,
            gaussian_likelihood=self.gaussian_likelihood,
            noise_model_likelihood=self.noise_model_likelihood,
        )

        # Logging
        # Rename val_loss dict
        loss = {"_".join(["val", k]): v for k, v in loss.items()}
        self.log_dict(loss, on_epoch=True, prog_bar=True)
        curr_psnr = self.compute_val_psnr(out, target)
        for i, psnr in enumerate(curr_psnr):
            self.log(f"val_psnr_ch{i+1}_batch", psnr, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        """Validation epoch end."""
        psnr_ = self.reduce_running_psnr()
        if psnr_ is not None:
            self.log("val_psnr", psnr_, on_epoch=True, prog_bar=True)
        else:
            self.log("val_psnr", 0.0, on_epoch=True, prog_bar=True)

    def predict_step(
        self, batch: tuple[ImageRegionData, ...], batch_idx: Any
    ) -> tuple[ImageRegionData, ImageRegionData]:
        """Prediction step.

        Runs `mmse_count` stochastic forward passes, returning the per-pixel mean
        (denormalized into target space) and the per-pixel sample standard deviation.

        Parameters
        ----------
        batch : tuple[ImageRegionData, ...]
            Input batch. The first element holds the input region; trailing elements
            (e.g. target) are ignored at prediction time.
        batch_idx : Any
            Batch index.

        Returns
        -------
        tuple[ImageRegionData, ImageRegionData]
            A pair `(mean_region, std_region)` carrying respectively the
            denormalized MMSE mean prediction and the per-pixel sample standard
            deviation. Both share the input region's metadata (source, axes,
            region_spec, ...) so that downstream tile stitching can be applied
            identically to either.
        """
        x = batch[0]
        # Reset model for inference with spatial dimensions only (H, W)
        self.model.reset_for_inference(x.data.shape[-2:]) # TODO: check this

        rec_img_list = []
        for _ in range(self.algorithm_config.mmse_count):
            rec, _ = self.model(x.data)

            # split out the predicted mean from logvar if the model emits both
            if self.model.predict_logvar is None:
                rec_img = rec
            else:
                rec_img, _ = torch.chunk(rec, chunks=2, dim=1)
            rec_img_list.append(rec_img.cpu().unsqueeze(0))  # add MMSE dim

        # aggregate over MMSE samples
        samples = torch.cat(rec_img_list, dim=0)
        mmse_imgs = torch.mean(samples, dim=0)
        std_imgs = torch.std(samples, dim=0)

        # Denormalize the MMSE mean using target-channel statistics owned by the
        # module (see `set_target_stats` / `on_predict_start`). The new
        # Normalization.denormalize() uses input stats only, so we build a
        # Denormalize transform manually.
        # TODO: clean this up once Normalization gains target-space denormalization.
        if self.target_means is None or self.target_stds is None:
            raise RuntimeError(
                "Target denormalization stats not set. Call "
                "`module.set_target_stats(target_means, target_stds)` before "
                "running `predict_step`, or run via `Trainer.predict(...)` with "
                "a data module whose predict_dataset.normalization exposes them."
            )
        denorm = Denormalize(
            image_means=self.target_means, image_stds=self.target_stds
        )
        mean_array = denorm(patch=mmse_imgs.numpy())
        std_array = std_imgs.numpy()

        # The input region's data_shape carries the *input* channel count (1 mixed
        # channel for MicroSplit), but the predicted tensor has `output_channels`
        # unmixed channels. Override the C dimension so downstream stitchers
        # (which allocate from `data_shape`) sized things correctly.
        output_data_shape = (
            int(x.data_shape[0]),
            int(self.algorithm_config.model.output_channels),
            *(int(d) for d in x.data_shape[2:]),
        )

        mean_region = ImageRegionData(
            data=mean_array,
            source=x.source,
            data_shape=output_data_shape,
            dtype=x.dtype,
            axes=x.axes,
            region_spec=x.region_spec,
            additional_metadata=x.additional_metadata,
            original_data_shape=x.original_data_shape,
        )
        std_region = ImageRegionData(
            data=std_array,
            source=x.source,
            data_shape=output_data_shape,
            dtype=x.dtype,
            axes=x.axes,
            region_spec=x.region_spec,
            additional_metadata=x.additional_metadata,
            original_data_shape=x.original_data_shape,
        )
        return mean_region, std_region

    # TODO use lightning.modules.model_utils configure_optimizers
    def configure_optimizers(self) -> Any:
        """Configure optimizers and learning rate schedulers.

        Returns
        -------
        Any
            Optimizer and learning rate scheduler.
        """
        # instantiate optimizer
        optimizer_func = get_optimizer(self.optimizer_name)
        optimizer = optimizer_func(self.model.parameters(), **self.optimizer_params)

        # and scheduler
        scheduler_func = get_scheduler(self.lr_scheduler_name)
        scheduler = scheduler_func(optimizer, **self.lr_scheduler_params)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",  # otherwise triggers MisconfigurationException
        }

    # TODO: find a way to move the following methods to a separate module
    # TODO: this same operation is done in many other places, like in loss_func
    # should we refactor LadderVAE so that it already outputs
    # tuple(`mean`, `logvar`, `td_data`)?
    def get_reconstructed_tensor(
        self, model_outputs: tuple[torch.Tensor, dict[str, Any]]
    ) -> torch.Tensor:
        """Get the reconstructed tensor from the LVAE model outputs.

        Parameters
        ----------
        model_outputs : tuple[torch.Tensor, dict[str, Any]]
            Model outputs. It is a tuple with a tensor representing the predicted mean
            and (optionally) logvar, and the top-down data dictionary.

        Returns
        -------
        torch.Tensor
            Reconstructed tensor, i.e., the predicted mean.
        """
        predictions, _ = model_outputs
        if self.model.predict_logvar is None:
            return predictions
        elif self.model.predict_logvar == "pixelwise":
            return predictions.chunk(2, dim=1)[0]

    def compute_val_psnr(
        self,
        model_output: tuple[torch.Tensor, dict[str, Any]],
        target: torch.Tensor,
        psnr_func: Callable = scale_invariant_psnr,
    ) -> list[float]:
        """Compute the PSNR for the current validation batch.

        Parameters
        ----------
        model_output : tuple[torch.Tensor, dict[str, Any]]
            Model output, a tuple with the predicted mean and (optionally) logvar,
            and the top-down data dictionary.
        target : torch.Tensor
            Target tensor.
        psnr_func : Callable, optional
            PSNR function to use, by default `scale_invariant_psnr`.

        Returns
        -------
        list[float]
            PSNR for each channel in the current batch.
        """
        # TODO check this! Related to is_supervised which is also wacky
        out_channels = target.shape[1]

        # get the reconstructed image
        recons_img = self.get_reconstructed_tensor(model_output)

        # update running psnr
        for i in range(out_channels):
            self.running_psnr[i].update(rec=recons_img[:, i], tar=target[:, i])

        # compute psnr for each channel in the current batch
        # TODO: this doesn't need do be a method of this class
        # and hence can be moved to a separate module
        return [
            psnr_func(
                gt=target[:, i].clone().detach().cpu().numpy(),
                pred=recons_img[:, i].clone().detach().cpu().numpy(),
            )
            for i in range(out_channels)
        ]

    def reduce_running_psnr(self) -> float | None:
        """Reduce the running PSNR statistics and reset the running PSNR.

        Returns
        -------
        Optional[float]
            Running PSNR averaged over the different output channels.
        """
        psnr_arr = []  # type: ignore
        for i in range(len(self.running_psnr)):
            psnr = self.running_psnr[i].get()
            if psnr is None:
                psnr_arr = None  # type: ignore
                break
            psnr_arr.append(psnr.cpu().numpy())
            self.running_psnr[i].reset()
            # TODO: this line forces it to be a method of this class
            # alternative is returning also the reset `running_psnr`
        if psnr_arr is not None:
            psnr = np.mean(psnr_arr)
        return psnr
