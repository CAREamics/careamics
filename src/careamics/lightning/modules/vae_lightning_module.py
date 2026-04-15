"""CAREamics Lightning module."""

from collections.abc import Callable
from typing import Any, Union

import numpy as np
import pytorch_lightning as L
import torch

from careamics.compat.transforms.normalize import Denormalize
from careamics.compat.transforms.tta import ImageRestorationTTA
from careamics.config import (
    VAEBasedAlgorithm,
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
from careamics.utils.torch_utils import get_optimizer, get_scheduler

NoiseModel = Union[GaussianMixtureNoiseModel, MultiChannelNoiseModel]

# TODO TTA and Denormalize are now in careamics.compat, need to investigate
# reimplementating them
# TODO Imported here for use by create_careamics_module


# TODO rename module
class VAEModule(L.LightningModule):
    """
    CAREamics Lightning module.

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

        # TODO: log algorithm config
        # self.save_hyperparameters(self.algorithm_config.model_dump())

        # create model
        self.model: torch.nn.Module = model_factory(self.algorithm_config.model)

        # supervised_mode
        self.supervised_mode = self.algorithm_config.is_supervised
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
        return self.model(x)  # TODO Different model can have more than one output

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

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: Any
    ) -> dict[str, torch.Tensor] | None:
        """Training step.

        Parameters
        ----------
        batch : tuple[torch.Tensor, torch.Tensor]
            Input batch. It is a tuple with the input tensor and the target tensor.
            The input tensor has shape (B, (1 + n_LC), [Z], Y, X), where n_LC is the
            number of lateral inputs. The target tensor has shape (B, C, [Z], Y, X),
            where C is the number of target channels (e.g., 1 in HDN, >1 in
            muSplit/denoiSplit).
        batch_idx : Any
            Batch index.

        Returns
        -------
        Any
            Loss value.
        """
        x, *target = batch

        # Forward pass
        out = self.model(x)
        if not self.supervised_mode:
            target = x
        else:
            target = target[
                0
            ]  # hacky way to unpack. #TODO maybe should be fixed on the dataset level

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
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: Any
    ) -> None:
        """Validation step.

        Parameters
        ----------
        batch : tuple[torch.Tensor, torch.Tensor]
            Input batch. It is a tuple with the input tensor and the target tensor.
            The input tensor has shape (B, (1 + n_LC), [Z], Y, X), where n_LC is the
            number of lateral inputs. The target tensor has shape (B, C, [Z], Y, X),
            where C is the number of target channels (e.g., 1 in HDN, >1 in
            muSplit/denoiSplit).
        batch_idx : Any
            Batch index.
        """
        x, *target = batch

        # Forward pass
        out = self.model(x)
        if not self.supervised_mode:
            target = x
        else:
            target = target[
                0
            ]  # hacky way to unpack. #TODO maybe should be fixed on the datasel level
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

    def predict_step(self, batch: torch.Tensor, batch_idx: Any) -> Any:
        """Prediction step.

        Parameters
        ----------
        batch : torch.Tensor
            Input batch.
        batch_idx : Any
            Batch index.

        Returns
        -------
        Any
            Model output.
        """
        if self.algorithm_config.algorithm == "microsplit":
            x, *aux = batch
            # Reset model for inference with spatial dimensions only (H, W)
            self.model.reset_for_inference(x.shape[-2:])

            rec_img_list = []
            for _ in range(self.algorithm_config.mmse_count):
                # get model output
                rec, _ = self.model(x)

                # get reconstructed img
                if self.model.predict_logvar is None:
                    rec_img = rec
                    _logvar = torch.tensor([-1])
                else:
                    rec_img, _logvar = torch.chunk(rec, chunks=2, dim=1)
                rec_img_list.append(rec_img.cpu().unsqueeze(0))  # add MMSE dim

            # aggregate results
            samples = torch.cat(rec_img_list, dim=0)
            mmse_imgs = torch.mean(samples, dim=0)  # avg over MMSE dim
            std_imgs = torch.std(samples, dim=0)  # std over MMSE dim

            # Denormalize the output using target channel statistics
            mean_dict, std_dict = (
                self._trainer.datamodule.predict_dataset.get_mean_std()
            )
            means_list = np.atleast_1d(mean_dict["target"].squeeze()).tolist()
            stds_list = np.atleast_1d(std_dict["target"].squeeze()).tolist()
            denorm = Denormalize(
                image_means=means_list,
                image_stds=stds_list,
            )
            tile_prediction = denorm(patch=mmse_imgs.cpu().numpy())
            tile_std = std_imgs.cpu().numpy()

            return tile_prediction, tile_std

        else:
            # Regular prediction logic
            if self._trainer.datamodule.tiled:
                # TODO tile_size should match model input size
                x, *aux = batch
                x = (
                    x[0] if isinstance(x, list | tuple) else x
                )  # TODO ugly, so far i don't know why x might be a list
                self.model.reset_for_inference(x.shape)  # TODO should it be here ?
            else:
                x = batch[0] if isinstance(batch, list | tuple) else batch
                aux = []
                self.model.reset_for_inference(x.shape)

            mmse_list = []
            for _ in range(self.algorithm_config.mmse_count):
                # apply test-time augmentation if available
                if self._trainer.datamodule.prediction_config.tta_transforms:
                    tta = ImageRestorationTTA()
                    augmented_batch = tta.forward(x)  # list of augmented tensors
                    augmented_output = []
                    for augmented in augmented_batch:
                        augmented_pred = self.model(augmented)
                        augmented_output.append(augmented_pred)
                    output = tta.backward(augmented_output)
                else:
                    output = self.model(x)

                # taking the 1st element of the output, 2nd is std if
                # predict_logvar=="pixelwise"
                output = (
                    output[0]
                    if self.model.predict_logvar is None
                    else output[0][:, 0:1, ...]
                )
                mmse_list.append(output)

            mmse = torch.stack(mmse_list).mean(0)
            std = torch.stack(mmse_list).std(0)  # TODO why?
            # TODO better way to unpack if pred logvar
            # Denormalize the output
            denorm = Denormalize(
                image_means=self._trainer.datamodule.predict_dataset.image_means,
                image_stds=self._trainer.datamodule.predict_dataset.image_stds,
            )

            denormalized_output = denorm(patch=mmse.cpu().numpy())

            if len(aux) > 0:  # aux can be tiling information
                return denormalized_output, std, *aux
            else:
                return denormalized_output, std

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
