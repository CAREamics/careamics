"""COSDD Lightning module.

Adapted from the COSDD reference implementation
(https://github.com/krulllab/COSDD, `models/hub.py`), MIT License,
Copyright (c) 2023 krulllab.
"""

from functools import partial
from typing import TYPE_CHECKING, Any, cast

import lightning.pytorch as L
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from careamics.config import COSDDAlgorithm
from careamics.dataset import ImageRegionData
from careamics.dataset.factory import TrainValData, TrainValSplitData
from careamics.dataset.normalization.normalization import Normalization
from careamics.models.cosdd import create_cosdd_networks
from careamics.utils.logging import get_logger

from .module_utils import (
    get_optimizer,
    get_scheduler,
    log_training_stats,
    mmse_and_sample_std,
)

logger = get_logger(__name__)

if TYPE_CHECKING:
    from careamics.lightning.data.data_module import CareamicsDataModule


class COSDDModule(L.LightningModule):
    """CAREamics PyTorch Lightning module for the COSDD algorithm.

    COSDD (COrrelated and Signal-Dependent Denoising) jointly trains:

    - a Ladder VAE encoding the signal into a latent code, together with an
      autoregressive decoder modelling the noisy image given the latent code (ELBO
      loss, logged as `train_loss`/`val_loss`),
    - a signal decoder mapping the latent code to image space (`sd_loss`),
    - an optional direct denoiser predicting the average of the signal samples
      (`dd_loss`).

    Each network is updated by its own optimizer, using manual optimization with
    gradient accumulation over `n_grad_batches` batches.

    At prediction, if `n_samples` is 1 and a direct denoiser was trained (and
    `direct_pred` is `True`), the direct denoiser is used. Otherwise, `n_samples`
    signal samples are drawn and averaged, and their standard deviation is returned
    as an uncertainty estimate if `n_samples > 1`.

    Parameters
    ----------
    algorithm_config : COSDDAlgorithm or dict
        Configuration for the COSDD algorithm, either as a `COSDDAlgorithm` instance
        or a dictionary.
    """

    def __init__(self, algorithm_config: COSDDAlgorithm | dict[str, Any]) -> None:
        """Instantiate COSDDModule.

        Parameters
        ----------
        algorithm_config : COSDDAlgorithm or dict
            Configuration for the COSDD algorithm, either as a `COSDDAlgorithm`
            instance or a dictionary.
        """
        super().__init__()

        if isinstance(algorithm_config, dict):
            config = COSDDAlgorithm(**algorithm_config)
        else:
            config = algorithm_config

        if not isinstance(config, COSDDAlgorithm):
            raise TypeError("algorithm_config must be a COSDDAlgorithm")

        self.save_hyperparameters({"algorithm_config": config.model_dump(mode="json")})
        self.config: COSDDAlgorithm = config

        networks = create_cosdd_networks(self.config.model)
        self.vae = networks.vae
        self.ar_decoder = networks.ar_decoder
        self.s_decoder = networks.s_decoder
        self.direct_denoiser = networks.direct_denoiser

        self.n_grad_batches: int = self.config.n_grad_batches
        self.automatic_optimization = False

        self.n_samples: int = 1
        """Number of signal samples averaged at prediction."""

        self.direct_pred: bool = self.direct_denoiser is not None
        """Whether to use the direct denoiser at prediction when `n_samples` is 1."""

    def on_fit_start(self) -> None:
        """On fit start hook for COSDD module."""
        assert self._trainer is not None
        datamodule: CareamicsDataModule = self._trainer.datamodule  # type: ignore[union-attr]
        assert isinstance(datamodule._data, (TrainValData, TrainValSplitData))
        if datamodule._data.train_data_target is not None:
            logger.warning(
                "COSDD is an unsupervised algorithm, arguments passed to "
                "`train_data_target` will be ignored.",
            )

    def vae_forward(self, x: torch.Tensor) -> dict[str, Any]:
        """Encode the input and predict the noise model parameters.

        Parameters
        ----------
        x : torch.Tensor
            Normalized noisy images, shape (B, C, [Z], Y, X).

        Returns
        -------
        dict of {str: Any}
            Latent signal code (`s_code`), approximate posteriors (`q_list`), priors
            (`p_list`) and autoregressive decoder output (`x_params`).
        """
        vae_out = self.vae(x)
        s_code = vae_out["s_code"]

        x_params = self.ar_decoder(x, s_code)
        out = {
            "s_code": s_code,
            "q_list": vae_out["q_list"],
            "p_list": vae_out["p_list"],
            "x_params": x_params,
        }
        return out

    def s_decoder_forward(self, s_code: torch.Tensor) -> torch.Tensor:
        """Decode the latent code into a signal estimate.

        The latent code is detached so that the signal decoder loss does not
        propagate to the VAE.

        Parameters
        ----------
        s_code : torch.Tensor
            Latent signal code.

        Returns
        -------
        torch.Tensor
            Signal estimate in normalized space.
        """
        s_hat = self.s_decoder(s_code.detach())
        return s_hat

    def direct_denoiser_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the denoised image with the direct denoiser.

        Parameters
        ----------
        x : torch.Tensor
            Normalized noisy images.

        Returns
        -------
        torch.Tensor
            Denoised images in normalized space.
        """
        assert self.direct_denoiser is not None
        s_direct = self.direct_denoiser(x)
        return s_direct

    def vae_loss(self, x: torch.Tensor, out: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Compute the negative ELBO, normalized per pixel.

        Parameters
        ----------
        x : torch.Tensor
            Normalized noisy images.
        out : dict of {str: Any}
            Output of `vae_forward`.

        Returns
        -------
        dict of {str: torch.Tensor}
            Negative ELBO (`elbo`), KL divergence (`kl_div`) and negative log
            likelihood (`nll`).
        """
        kl_div = self.vae.kl_divergence(out["q_list"], out["p_list"])
        kl_div = kl_div / x.numel()
        nll = -self.ar_decoder.loglikelihood(x, out["x_params"]).mean()
        elbo = kl_div + nll
        return {"elbo": elbo, "kl_div": kl_div, "nll": nll}

    def s_decoder_loss(self, x: torch.Tensor, s_hat: torch.Tensor) -> torch.Tensor:
        """Compute the signal decoder loss.

        Parameters
        ----------
        x : torch.Tensor
            Normalized noisy images.
        s_hat : torch.Tensor
            Signal estimate.

        Returns
        -------
        torch.Tensor
            Mean squared error.
        """
        return self.s_decoder.loss(x, s_hat).mean()

    def direct_denoiser_loss(
        self, s_hat: torch.Tensor, s_direct: torch.Tensor
    ) -> torch.Tensor:
        """Compute the direct denoiser loss against the signal decoder estimate.

        Parameters
        ----------
        s_hat : torch.Tensor
            Signal estimate of the signal decoder, used as target.
        s_direct : torch.Tensor
            Direct denoiser estimate.

        Returns
        -------
        torch.Tensor
            Mean L1 or squared error.
        """
        assert self.direct_denoiser is not None
        return self.direct_denoiser.loss(s_hat.detach(), s_direct).mean()

    def training_step(
        self,
        batch: tuple[ImageRegionData] | tuple[ImageRegionData, ImageRegionData],
        batch_idx: int,
    ) -> None:
        """Training step for COSDD.

        Parameters
        ----------
        batch : ImageRegionData or (ImageRegionData, ImageRegionData)
            A tuple containing the input data and, optionally, ignored target data.
        batch_idx : int
            The index of the current batch in the training loop.
        """
        x = cast(torch.Tensor, batch[0].data)
        batch_size = x.shape[0]
        optimizers = cast(list[Any], self.optimizers())
        update = (batch_idx + 1) % self.n_grad_batches == 0

        vae_out = self.vae_forward(x)
        vae_losses = self.vae_loss(x, vae_out)
        self.manual_backward(vae_losses["elbo"])
        log_training_stats(self, vae_losses["elbo"], batch_size=batch_size)
        self.log(
            "train_kl",
            vae_losses["kl_div"],
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train_nll",
            vae_losses["nll"],
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        if update:
            optimizers[0].step()
            optimizers[0].zero_grad()

        s_hat = self.s_decoder_forward(vae_out["s_code"])
        sd_loss = self.s_decoder_loss(x, s_hat)
        self.manual_backward(sd_loss)
        self.log(
            "train_sd_loss",
            sd_loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        if update:
            optimizers[1].step()
            optimizers[1].zero_grad()

        if self.direct_denoiser is not None:
            s_direct = self.direct_denoiser_forward(x)
            dd_loss = self.direct_denoiser_loss(s_hat, s_direct)
            self.manual_backward(dd_loss)
            self.log(
                "train_dd_loss",
                dd_loss,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
            if update:
                optimizers[2].step()
                optimizers[2].zero_grad()

    def on_train_epoch_end(self) -> None:
        """Step the learning rate schedulers.

        `ReduceLROnPlateau` schedulers monitor the validation loss of their network
        and are not stepped if it has not been computed.
        """
        schedulers = cast(list[Any], self.lr_schedulers())
        monitored = ["val_loss", "val_sd_loss", "val_dd_loss"]
        for scheduler, metric in zip(schedulers, monitored, strict=False):
            if isinstance(scheduler, ReduceLROnPlateau):
                value = self.trainer.callback_metrics.get(metric)
                if value is not None:
                    scheduler.step(value)
            else:
                scheduler.step()

    def validation_step(
        self,
        batch: tuple[ImageRegionData] | tuple[ImageRegionData, ImageRegionData],
        batch_idx: int,
    ) -> None:
        """Validation step for COSDD.

        Parameters
        ----------
        batch : ImageRegionData or (ImageRegionData, ImageRegionData)
            A tuple containing the input data and, optionally, ignored target data.
        batch_idx : int
            The index of the current batch in the validation loop.
        """
        x = cast(torch.Tensor, batch[0].data)
        batch_size = x.shape[0]

        vae_out = self.vae_forward(x)
        vae_losses = self.vae_loss(x, vae_out)
        self.log(
            "val_loss",
            vae_losses["elbo"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log("val_kl", vae_losses["kl_div"], batch_size=batch_size)
        self.log("val_nll", vae_losses["nll"], batch_size=batch_size)

        s_hat = self.s_decoder_forward(vae_out["s_code"])
        sd_loss = self.s_decoder_loss(x, s_hat)
        self.log("val_sd_loss", sd_loss, batch_size=batch_size)

        if self.direct_denoiser is not None:
            s_direct = self.direct_denoiser_forward(x)
            dd_loss = self.direct_denoiser_loss(s_hat, s_direct)
            self.log("val_dd_loss", dd_loss, batch_size=batch_size)

    def predict_sample(
        self, x_data: torch.Tensor, normalization: Normalization
    ) -> torch.Tensor:
        """Draw one signal sample and denormalize it.

        Parameters
        ----------
        x_data : torch.Tensor
            Normalized noisy images, shape (B, C, [Z], Y, X).
        normalization : Normalization
            Normalization used to map the sample back to the data space.

        Returns
        -------
        torch.Tensor
            A single signal sample, shape (B, C, [Z], Y, X).
        """
        s_code = self.vae(x_data)["s_code"]
        s_hat = self.s_decoder(s_code)
        return normalization.denormalize(s_hat.float())

    def predict_step(
        self,
        batch: tuple[ImageRegionData] | tuple[ImageRegionData, ImageRegionData],
        batch_idx: int,
    ) -> tuple[ImageRegionData, ImageRegionData | None]:
        """Prediction step for COSDD.

        Parameters
        ----------
        batch : ImageRegionData or (ImageRegionData, ImageRegionData)
            A tuple containing the input data and, optionally, the target data.
        batch_idx : int
            The index of the current batch in the prediction loop.

        Returns
        -------
        tuple of (ImageRegionData, ImageRegionData or None)
            The denoised batch and, if several samples were drawn, their standard
            deviation.
        """
        x = batch[0]
        x_data = cast(torch.Tensor, x.data)
        normalization = self._trainer.datamodule.predict_dataset.normalization  # type: ignore[union-attr]

        std: torch.Tensor | None
        if self.n_samples == 1 and self.direct_pred:
            if self.direct_denoiser is None:
                raise ValueError(
                    "No direct denoiser was trained, set `direct_pred` to False to "
                    "average signal samples instead."
                )
            s_direct = self.direct_denoiser(x_data)
            mean, std = normalization.denormalize(s_direct.float()), None
        else:
            mean, std = mmse_and_sample_std(
                partial(self.predict_sample, normalization=normalization),
                x_data,
                self.n_samples,
            )

        prediction = ImageRegionData.from_model_output(x, mean.cpu().numpy())
        if std is None:
            return prediction, None
        uncertainty = prediction._replace(data=std.cpu().numpy())
        return prediction, uncertainty

    @torch.no_grad()
    def sample_prior(
        self, n_imgs: int, normalization: Normalization | None = None
    ) -> dict[str, torch.Tensor]:
        """Generate new images by sampling from the prior.

        Images have the spatial shape of the training patches.

        Parameters
        ----------
        n_imgs : int
            Number of images to generate.
        normalization : Normalization or None, default=None
            If provided, used to map the generated images to the data space.
            Otherwise, they are returned in normalized space.

        Returns
        -------
        dict of {str: torch.Tensor}
            Generated noisy images (`x`) and their clean signal (`s`).
        """
        self.eval()
        s_code = self.vae.sample_from_prior(n_imgs)
        s = self.s_decoder(s_code)
        x = self.ar_decoder.sample(s_code)
        if normalization is not None:
            s = normalization.denormalize(s)
            x = normalization.denormalize(x)

        return {"x": x, "s": s}

    @torch.no_grad()
    def reconstruct(
        self, x: torch.Tensor, normalization: Normalization | None = None
    ) -> dict[str, torch.Tensor]:
        """Sample a signal and a noisy reconstruction of normalized noisy images.

        Parameters
        ----------
        x : torch.Tensor
            Normalized noisy images, shape (B, C, [Z], Y, X).
        normalization : Normalization or None, default=None
            If provided, used to map the outputs to the data space. Otherwise, they
            are returned in normalized space.

        Returns
        -------
        dict of {str: torch.Tensor}
            Noisy reconstruction (`x_hat`) and signal sample (`s_hat`).
        """
        self.eval()
        s_code = self.vae(x)["s_code"]
        s_hat = self.s_decoder(s_code)
        x_hat = self.ar_decoder.sample(s_code)
        if normalization is not None:
            s_hat = normalization.denormalize(s_hat)
            x_hat = normalization.denormalize(x_hat)

        return {"x_hat": x_hat, "s_hat": s_hat}

    def configure_optimizers(self) -> tuple[list[Any], list[Any]]:  # type: ignore[override]
        """Configure one optimizer and learning rate scheduler per network.

        Returns
        -------
        tuple of (list, list)
            Optimizers and schedulers of the VAE with the autoregressive decoder,
            the signal decoder and, if trained, the direct denoiser.
        """
        param_groups = [
            list(self.vae.parameters()) + list(self.ar_decoder.parameters()),
            list(self.s_decoder.parameters()),
        ]
        if self.direct_denoiser is not None:
            param_groups.append(list(self.direct_denoiser.parameters()))

        optimizer_cls = get_optimizer(self.config.optimizer.name)
        scheduler_cls = get_scheduler(self.config.lr_scheduler.name)

        optimizers = []
        schedulers = []
        for params in param_groups:
            optimizer = optimizer_cls(  # type: ignore[call-arg]
                params, **self.config.optimizer.parameters
            )
            optimizers.append(optimizer)
            schedulers.append(
                scheduler_cls(  # type: ignore[call-arg]
                    optimizer, **self.config.lr_scheduler.parameters
                )
            )

        return optimizers, schedulers
