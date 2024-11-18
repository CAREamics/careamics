"""Methods for Loss Computation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import numpy as np
import torch

from careamics.losses.lvae.loss_utils import free_bits_kl, get_kl_weight
from careamics.models.lvae.likelihoods import (
    GaussianLikelihood,
    LikelihoodModule,
    NoiseModelLikelihood,
)

if TYPE_CHECKING:
    from careamics.config import LVAELossConfig

Likelihood = Union[LikelihoodModule, GaussianLikelihood, NoiseModelLikelihood]


def get_reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    likelihood_obj: Likelihood,
) -> dict[str, torch.Tensor]:
    """Compute the reconstruction loss (negative log-likelihood).

    Parameters
    ----------
    reconstruction: torch.Tensor
        The output of the LVAE decoder. Shape is (B, C, [Z], Y, X), where C is the
        number of output channels (e.g., 1 in HDN, >1 in muSplit/denoiSplit).
    target: torch.Tensor
        The target image used to compute the reconstruction loss. Shape is
        (B, C, [Z], Y, X), where C is the number of output channels
        (e.g., 1 in HDN, >1 in muSplit/denoiSplit).
    likelihood_obj: Likelihood
        The likelihood object used to compute the reconstruction loss.

    Returns
    -------
    torch.Tensor
        The recontruction loss (negative log-likelihood).
    """
    # Compute Log likelihood
    ll, _ = likelihood_obj(reconstruction, target)  # shape: (B, C, [Z], Y, X)
    return -1 * ll.mean()


def _reconstruction_loss_musplit_denoisplit(
    predictions: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
    targets: torch.Tensor,
    nm_likelihood: NoiseModelLikelihood,
    gaussian_likelihood: GaussianLikelihood,
    nm_weight: float,
    gaussian_weight: float,
) -> torch.Tensor:
    """Compute the reconstruction loss for muSplit-denoiSplit loss.

    The resulting loss is a weighted mean of the noise model likelihood and the
    Gaussian likelihood.

    Parameters
    ----------
    predictions : torch.Tensor
        The output of the LVAE decoder. Shape is (B, C, [Z], Y, X), or
        (B, 2*C, [Z], Y, X), where C is the number of output channels,
        and the factor of 2 is for the case of predicted log-variance.
    targets : torch.Tensor
        The target image used to compute the reconstruction loss. Shape is
        (B, C, [Z], Y, X), where C is the number of output channels
        (e.g., 1 in HDN, >1 in muSplit/denoiSplit).
    nm_likelihood : NoiseModelLikelihood
        A `NoiseModelLikelihood` object used to compute the noise model likelihood.
    gaussian_likelihood : GaussianLikelihood
        A `GaussianLikelihood` object used to compute the Gaussian likelihood.
    nm_weight : float
        The weight for the noise model likelihood.
    gaussian_weight : float
        The weight for the Gaussian likelihood.

    Returns
    -------
    recons_loss : torch.Tensor
        The reconstruction loss. Shape is (1, ).
    """
    if predictions.shape[1] == 2 * targets.shape[1]:
        # predictions contain both mean and log-variance
        pred_mean, _ = predictions.chunk(2, dim=1)
    else:
        pred_mean = predictions

    recons_loss_nm = get_reconstruction_loss(
        reconstruction=pred_mean, target=targets, likelihood_obj=nm_likelihood
    )

    recons_loss_gm = get_reconstruction_loss(
        reconstruction=predictions,
        target=targets,
        likelihood_obj=gaussian_likelihood,
    )

    recons_loss = nm_weight * recons_loss_nm + gaussian_weight * recons_loss_gm
    return recons_loss


def get_kl_divergence_loss(
    kl_type: Literal["kl", "kl_restricted"],
    topdown_data: dict[str, torch.Tensor],
    rescaling: Literal["latent_dim", "image_dim"],
    aggregation: Literal["mean", "sum"],
    free_bits_coeff: float,
    img_shape: Optional[tuple[int]] = None,
) -> torch.Tensor:
    """Compute the KL divergence loss.

    NOTE: Description of `rescaling` methods:
    - If "latent_dim", the KL-loss values are rescaled w.r.t. the latent space
    dimensions (spatial + number of channels, i.e., (C, [Z], Y, X)). In this way they
    have the same magnitude across layers.
    - If "image_dim", the KL-loss values are rescaled w.r.t. the input image spatial
    dimensions. In this way, the lower layers have a larger KL-loss value compared to
    the higher layers, since the latent space and hence the KL tensor has more entries.
    Specifically, at hierarchy `i`, the total KL loss is larger by a factor (128/i**2).

    NOTE: the type of `aggregation` determines the magnitude of the KL-loss. Clearly,
    "sum" aggregation results in a larger KL-loss value compared to "mean" by a factor
    of `n_layers`.

    NOTE: recall that sample-wise KL is obtained by summing over all dimensions,
    including Z. Also recall that in current 3D implementation of LVAE, no downsampling
    is done on Z. Therefore, to avoid emphasizing KL loss too much, we divide it
    by the Z dimension of input image in every case.

    Parameters
    ----------
    kl_type : Literal["kl", "kl_restricted"]
        The type of KL divergence loss to compute.
    topdown_data : dict[str, torch.Tensor]
        A dictionary containing information computed for each layer during the top-down
        pass. The dictionary must include the following keys:
        - "kl": The KL-loss values for each layer. Shape of each tensor is (B,).
        - "z": The sampled latents for each layer. Shape of each tensor is
        (B, layers, `z_dims[i]`, H, W).
    rescaling : Literal["latent_dim", "image_dim"]
        The rescaling method used for the KL-loss values. If "latent_dim", the KL-loss
        values are rescaled w.r.t. the latent space dimensions (spatial + number of
        channels, i.e., (C, [Z], Y, X)). If "image_dim", the KL-loss values are
        rescaled w.r.t. the input image spatial dimensions.
    aggregation : Literal["mean", "sum"]
        The aggregation method used to combine the KL-loss values across layers. If
        "mean", the KL-loss values are averaged across layers. If "sum", the KL-loss
        values are summed across layers.
    free_bits_coeff : float
        The free bits coefficient used for the KL-loss computation.
    img_shape : Optional[tuple[int]]
        The shape of the input image to the LVAE model. Shape is ([Z], Y, X).

    Returns
    -------
    kl_loss : torch.Tensor
        The KL divergence loss. Shape is (1, ).
    """
    kl = torch.cat(
        [kl_layer.unsqueeze(1) for kl_layer in topdown_data[kl_type]],
        dim=1,
    )  # shape: (B, n_layers)

    # Apply free bits (& batch average)
    kl = free_bits_kl(kl, free_bits_coeff)  # shape: (n_layers,)

    # In 3D case, rescale by Z dim
    # TODO If we have downsampling in Z dimension, then this needs to change.
    if len(img_shape) == 3:
        kl = kl / img_shape[0]

    # Rescaling
    if rescaling == "latent_dim":
        for i in range(len(kl)):
            latent_dim = topdown_data["z"][i].shape[1:]
            norm_factor = np.prod(latent_dim)
            kl[i] = kl[i] / norm_factor
    elif rescaling == "image_dim":
        kl = kl / np.prod(img_shape[-2:])

    # Aggregation
    if aggregation == "mean":
        kl = kl.mean()  # shape: (1,)
    elif aggregation == "sum":
        kl = kl.sum()  # shape: (1,)

    return kl


def _get_kl_divergence_loss_musplit(
    topdown_data: dict[str, torch.Tensor],
    img_shape: tuple[int],
    kl_type: Literal["kl", "kl_restricted"],
) -> torch.Tensor:
    """Compute the KL divergence loss for muSplit.

    Parameters
    ----------
    topdown_data : dict[str, torch.Tensor]
        A dictionary containing information computed for each layer during the top-down
        pass. The dictionary must include the following keys:
        - "kl": The KL-loss values for each layer. Shape of each tensor is (B,).
        - "z": The sampled latents for each layer. Shape of each tensor is
        (B, layers, `z_dims[i]`, H, W).
    img_shape : tuple[int]
        The shape of the input image to the LVAE model. Shape is ([Z], Y, X).
    kl_type : Literal["kl", "kl_restricted"]
        The type of KL divergence loss to compute.

    Returns
    -------
    kl_loss : torch.Tensor
        The KL divergence loss for the muSplit case. Shape is (1, ).
    """
    return get_kl_divergence_loss(
        kl_type="kl",  # TODO: hardcoded, deal in future PR
        topdown_data=topdown_data,
        rescaling="latent_dim",
        aggregation="mean",
        free_bits_coeff=0.0,
        img_shape=img_shape,
    )


def _get_kl_divergence_loss_denoisplit(
    topdown_data: dict[str, torch.Tensor],
    img_shape: tuple[int],
    kl_type: Literal["kl", "kl_restricted"],
) -> torch.Tensor:
    """Compute the KL divergence loss for denoiSplit.

    Parameters
    ----------
    topdown_data : dict[str, torch.Tensor]
        A dictionary containing information computed for each layer during the top-down
        pass. The dictionary must include the following keys:
        - "kl": The KL-loss values for each layer. Shape of each tensor is (B,).
        - "z": The sampled latents for each layer. Shape of each tensor is
        (B, layers, `z_dims[i]`, H, W).
    img_shape : tuple[int]
        The shape of the input image to the LVAE model. Shape is ([Z], Y, X).
    kl_type : Literal["kl", "kl_restricted"]
        The type of KL divergence loss to compute.

    Returns
    -------
    kl_loss : torch.Tensor
        The KL divergence loss for the denoiSplit case. Shape is (1, ).
    """
    return get_kl_divergence_loss(
        kl_type=kl_type,
        topdown_data=topdown_data,
        rescaling="image_dim",
        aggregation="sum",
        free_bits_coeff=1.0,
        img_shape=img_shape,
    )


# TODO: @melisande-c suggested to refactor this as a class (see PR #208)
# - loss computation happens by calling the `__call__` method
# - `__init__` method initializes the loss parameters now contained in
# the `LVAELossParameters` class
# NOTE: same for the other loss functions
def musplit_loss(
    model_outputs: tuple[torch.Tensor, dict[str, Any]],
    targets: torch.Tensor,
    config: LVAELossConfig,
    gaussian_likelihood: Optional[GaussianLikelihood],
    noise_model_likelihood: Optional[NoiseModelLikelihood] = None,  # TODO: ugly
) -> Optional[dict[str, torch.Tensor]]:
    """Loss function for muSplit.

    Parameters
    ----------
    model_outputs : tuple[torch.Tensor, dict[str, Any]]
        Tuple containing the model predictions (shape is (B, `target_ch`, [Z], Y, X))
        and the top-down layer data (e.g., sampled latents, KL-loss values, etc.).
    targets : torch.Tensor
        The target image used to compute the reconstruction loss. Shape is
        (B, `target_ch`, [Z], Y, X).
    config : LVAELossConfig
        The config for loss function (e.g., KL hyperparameters, likelihood module,
        noise model, etc.).
    gaussian_likelihood : GaussianLikelihood
        The Gaussian likelihood object.
    noise_model_likelihood : Optional[NoiseModelLikelihood]
        The noise model likelihood object. Not used here.

    Returns
    -------
    output : Optional[dict[str, torch.Tensor]]
        A dictionary containing the overall loss `["loss"]`, the reconstruction loss
        `["reconstruction_loss"]`, and the KL divergence loss `["kl_loss"]`.
    """
    assert gaussian_likelihood is not None

    predictions, td_data = model_outputs

    # Reconstruction loss computation
    recons_loss = config.reconstruction_weight * get_reconstruction_loss(
        reconstruction=predictions,
        target=targets,
        likelihood_obj=gaussian_likelihood,
    )
    if torch.isnan(recons_loss).any():
        recons_loss = 0.0

    # KL loss computation
    kl_weight = get_kl_weight(
        config.kl_params.annealing,
        config.kl_params.start,
        config.kl_params.annealtime,
        config.kl_weight,
        config.kl_params.current_epoch,
    )
    kl_loss = (
        _get_kl_divergence_loss_musplit(
            topdown_data=td_data,
            img_shape=targets.shape[2:],
            kl_type=config.kl_params.loss_type,
        )
        * kl_weight
    )

    net_loss = recons_loss + kl_loss
    output = {
        "loss": net_loss,
        "reconstruction_loss": (
            recons_loss.detach()
            if isinstance(recons_loss, torch.Tensor)
            else recons_loss
        ),
        "kl_loss": kl_loss.detach(),
    }
    # https://github.com/openai/vdvae/blob/main/train.py#L26
    if torch.isnan(net_loss).any():
        return None

    return output


def denoisplit_loss(
    model_outputs: tuple[torch.Tensor, dict[str, Any]],
    targets: torch.Tensor,
    config: LVAELossConfig,
    gaussian_likelihood: Optional[GaussianLikelihood] = None,
    noise_model_likelihood: Optional[NoiseModelLikelihood] = None,
) -> Optional[dict[str, torch.Tensor]]:
    """Loss function for DenoiSplit.

    Parameters
    ----------
    model_outputs : tuple[torch.Tensor, dict[str, Any]]
        Tuple containing the model predictions (shape is (B, `target_ch`, [Z], Y, X))
        and the top-down layer data (e.g., sampled latents, KL-loss values, etc.).
    targets : torch.Tensor
        The target image used to compute the reconstruction loss. Shape is
        (B, `target_ch`, [Z], Y, X).
    config : LVAELossConfig
        The config for loss function containing all loss hyperparameters.
    gaussian_likelihood : GaussianLikelihood
        The Gaussian likelihood object.
    noise_model_likelihood : NoiseModelLikelihood
        The noise model likelihood object.

    Returns
    -------
    output : Optional[dict[str, torch.Tensor]]
        A dictionary containing the overall loss `["loss"]`, the reconstruction loss
        `["reconstruction_loss"]`, and the KL divergence loss `["kl_loss"]`.
    """
    assert noise_model_likelihood is not None

    predictions, td_data = model_outputs

    # Reconstruction loss computation
    recons_loss = config.reconstruction_weight * get_reconstruction_loss(
        reconstruction=predictions,
        target=targets,
        likelihood_obj=noise_model_likelihood,
    )
    if torch.isnan(recons_loss).any():
        recons_loss = 0.0

    # KL loss computation
    kl_weight = get_kl_weight(
        config.kl_params.annealing,
        config.kl_params.start,
        config.kl_params.annealtime,
        config.kl_weight,
        config.kl_params.current_epoch,
    )
    kl_loss = (
        _get_kl_divergence_loss_denoisplit(
            topdown_data=td_data,
            img_shape=targets.shape[2:],
            kl_type=config.kl_params.loss_type,
        )
        * kl_weight
    )

    net_loss = recons_loss + kl_loss
    output = {
        "loss": net_loss,
        "reconstruction_loss": (
            recons_loss.detach()
            if isinstance(recons_loss, torch.Tensor)
            else recons_loss
        ),
        "kl_loss": kl_loss.detach(),
    }
    # https://github.com/openai/vdvae/blob/main/train.py#L26
    if torch.isnan(net_loss).any():
        return None

    return output


def denoisplit_musplit_loss(
    model_outputs: tuple[torch.Tensor, dict[str, Any]],
    targets: torch.Tensor,
    config: LVAELossConfig,
    gaussian_likelihood: GaussianLikelihood,
    noise_model_likelihood: NoiseModelLikelihood,
) -> Optional[dict[str, torch.Tensor]]:
    """Loss function for DenoiSplit.

    Parameters
    ----------
    model_outputs : tuple[torch.Tensor, dict[str, Any]]
        Tuple containing the model predictions (shape is (B, `target_ch`, [Z], Y, X))
        and the top-down layer data (e.g., sampled latents, KL-loss values, etc.).
    targets : torch.Tensor
        The target image used to compute the reconstruction loss. Shape is
        (B, `target_ch`, [Z], Y, X).
    config : LVAELossConfig
        The config for loss function containing all loss hyperparameters.
    gaussian_likelihood : GaussianLikelihood
        The Gaussian likelihood object.
    noise_model_likelihood : NoiseModelLikelihood
        The noise model likelihood object.

    Returns
    -------
    output : Optional[dict[str, torch.Tensor]]
        A dictionary containing the overall loss `["loss"]`, the reconstruction loss
        `["reconstruction_loss"]`, and the KL divergence loss `["kl_loss"]`.
    """
    predictions, td_data = model_outputs

    # Reconstruction loss computation
    recons_loss = _reconstruction_loss_musplit_denoisplit(
        predictions=predictions,
        targets=targets,
        nm_likelihood=noise_model_likelihood,
        gaussian_likelihood=gaussian_likelihood,
        nm_weight=config.denoisplit_weight,
        gaussian_weight=config.musplit_weight,
    )
    if torch.isnan(recons_loss).any():
        recons_loss = 0.0

    # KL loss computation
    # NOTE: 'kl' key stands for the 'kl_samplewise' key in the TopDownLayer class.
    # The different naming comes from `top_down_pass()` method in the LadderVAE.
    denoisplit_kl = _get_kl_divergence_loss_denoisplit(
        topdown_data=td_data,
        img_shape=targets.shape[2:],
        kl_type=config.kl_params.loss_type,
    )
    musplit_kl = _get_kl_divergence_loss_musplit(
        topdown_data=td_data,
        img_shape=targets.shape[2:],
        kl_type=config.kl_params.loss_type,
    )
    kl_loss = (
        config.denoisplit_weight * denoisplit_kl + config.musplit_weight * musplit_kl
    )
    # TODO `kl_weight` is hardcoded (???)
    kl_loss = config.kl_weight * kl_loss

    net_loss = recons_loss + kl_loss
    output = {
        "loss": net_loss,
        "reconstruction_loss": (
            recons_loss.detach()
            if isinstance(recons_loss, torch.Tensor)
            else recons_loss
        ),
        "kl_loss": kl_loss.detach(),
    }
    # https://github.com/openai/vdvae/blob/main/train.py#L26
    if torch.isnan(net_loss).any():
        return None

    return output
