"""Methods for Loss Computation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Union

import numpy as np
import torch

from careamics.losses.lvae.loss_utils import free_bits_kl, get_kl_weight

if TYPE_CHECKING:
    from careamics.config import LVAELossConfig
    from careamics.models.lvae.noise_models import MultiChannelNoiseModel


def _compute_gaussian_log_likelihood(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    predict_logvar: bool,
    logvar_lowerbound: float | None = None,
) -> torch.Tensor:
    """Compute Gaussian log-likelihood.

    Parameters
    ----------
    reconstruction : torch.Tensor
        The output of the LVAE decoder. Shape is (B, C, [Z], Y, X) if predict_logvar
        is False, or (B, 2*C, [Z], Y, X) if predict_logvar is True.
    target : torch.Tensor
        The target image. Shape is (B, C, [Z], Y, X).
    predict_logvar : bool
        Whether log-variance is predicted pixel-wise.
    logvar_lowerbound : float | None, optional
        Lower bound for log-variance, by default None.

    Returns
    -------
    torch.Tensor
        The log-likelihood value (scalar).
    """
    if predict_logvar:
        mean, logvar = reconstruction.chunk(2, dim=1)
        if logvar_lowerbound is not None:
            logvar = torch.clip(logvar, min=logvar_lowerbound)
        var = torch.exp(logvar)
        log_prob = -0.5 * (
            ((target - mean) ** 2) / var + logvar + torch.tensor(2 * np.pi).log()
        )
    else:
        log_prob = -0.5 * (reconstruction - target) ** 2
    
    return log_prob.mean()


def _compute_noise_model_log_likelihood(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    noise_model: "MultiChannelNoiseModel",
    data_mean: float,
    data_std: float,
) -> torch.Tensor:
    """Compute noise model log-likelihood.

    Parameters
    ----------
    reconstruction : torch.Tensor
        The output of the LVAE decoder (normalized). Shape is (B, C, [Z], Y, X).
    target : torch.Tensor
        The target image (normalized). Shape is (B, C, [Z], Y, X).
    noise_model : MultiChannelNoiseModel
        The noise model to use for computing likelihood.
    data_mean : float
        Mean used for normalization (for denormalization).
    data_std : float
        Standard deviation used for normalization (for denormalization).

    Returns
    -------
    torch.Tensor
        The log-likelihood value (scalar).
    """
    # Convert to tensors and move to correct device
    data_mean_tensor = torch.as_tensor(data_mean, dtype=torch.float32, device=reconstruction.device)
    data_std_tensor = torch.as_tensor(data_std, dtype=torch.float32, device=reconstruction.device)
    
    # Move noise model to correct device if needed
    if reconstruction.device != noise_model.device:
        noise_model.to_device(reconstruction.device)
    
    # Denormalize predictions and targets
    reconstruction_denorm = reconstruction * data_std_tensor + data_mean_tensor
    target_denorm = target * data_std_tensor + data_mean_tensor
    
    # Compute likelihood using noise model
    likelihoods = noise_model.likelihood(target_denorm, reconstruction_denorm)
    log_prob = torch.log(likelihoods)
    
    return log_prob.mean()


def get_reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    likelihood_obj: Any,
) -> dict[str, torch.Tensor]:
    """Compute the reconstruction loss (negative log-likelihood).
    
    DEPRECATED: This function is kept for backward compatibility but should not be used.
    Use _compute_gaussian_log_likelihood or _compute_noise_model_log_likelihood instead.

    Parameters
    ----------
    reconstruction: torch.Tensor
        The output of the LVAE decoder. Shape is (B, C, [Z], Y, X), where C is the
        number of output channels (e.g., 1 in HDN, >1 in muSplit/denoiSplit).
    target: torch.Tensor
        The target image used to compute the reconstruction loss. Shape is
        (B, C, [Z], Y, X), where C is the number of output channels
        (e.g., 1 in HDN, >1 in muSplit/denoiSplit).
    likelihood_obj: Any
        The likelihood object used to compute the reconstruction loss.

    Returns
    -------
    torch.Tensor
        The recontruction loss (negative log-likelihood).
    """
    # Compute Log likelihood
    ll, _ = likelihood_obj(reconstruction, target)  # shape: (B, C, [Z], Y, X)
    return -1 * ll.mean()


def get_kl_divergence_loss(
    kl_type: Literal["kl", "kl_restricted"],
    topdown_data: dict[str, torch.Tensor],
    rescaling: Literal["latent_dim", "image_dim"],
    aggregation: Literal["mean", "sum"],
    free_bits_coeff: float,
    img_shape: tuple[int] | None = None,
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




# TODO: @melisande-c suggested to refactor this as a class (see PR #208)
# - loss computation happens by calling the `__call__` method
# - `__init__` method initializes the loss parameters now contained in
# the `LVAELossParameters` class
# NOTE: same for the other loss functions


def hdn_loss(
    model_outputs: tuple[torch.Tensor, dict[str, Any]],
    targets: torch.Tensor,
    config: LVAELossConfig,
    noise_model: "MultiChannelNoiseModel | None" = None,
    data_mean: float | None = None,
    data_std: float | None = None,
) -> dict[str, torch.Tensor] | None:
    """Loss function for HDN.

    Parameters
    ----------
    model_outputs : tuple[torch.Tensor, dict[str, Any]]
        Tuple containing the model predictions (shape is (B, `target_ch`, [Z], Y, X))
        and the top-down layer data (e.g., sampled latents, KL-loss values, etc.).
    targets : torch.Tensor
        The target image used to compute the reconstruction loss. In this case we use
        the input patch itself as target. Shape is (B, `target_ch`, [Z], Y, X).
    config : LVAELossConfig
        The config for loss function containing all loss hyperparameters.
    noise_model : MultiChannelNoiseModel | None, optional
        The noise model. Required if using noise model likelihood.
    data_mean : float | None, optional
        Data mean for denormalization. Required if using noise model.
    data_std : float | None, optional
        Data std for denormalization. Required if using noise model.

    Returns
    -------
    output : Optional[dict[str, torch.Tensor]]
        A dictionary containing the overall loss `["loss"]`, the reconstruction loss
        `["reconstruction_loss"]`, and the KL divergence loss `["kl_loss"]`.
    """
    predictions, td_data = model_outputs

    # Reconstruction loss computation
    # HDN can use either Gaussian or noise model likelihood
    if noise_model is not None and data_mean is not None and data_std is not None:
        recons_loss = config.reconstruction_weight * -_compute_noise_model_log_likelihood(
            reconstruction=predictions,
            target=targets,
            noise_model=noise_model,
            data_mean=data_mean,
            data_std=data_std,
        )
    else:
        recons_loss = config.reconstruction_weight * -_compute_gaussian_log_likelihood(
            reconstruction=predictions,
            target=targets,
            predict_logvar=config.predict_logvar,
            logvar_lowerbound=config.logvar_lowerbound,
        )
    if torch.isnan(recons_loss).any():
        recons_loss = 0.0

    # KL loss computation
    # Annealing is disabled, so kl_weight is just config.kl_weight
    kl_loss = (
        get_kl_divergence_loss(
            kl_type="kl_restricted",
            topdown_data=td_data,
            rescaling="image_dim",
            aggregation="sum",
            free_bits_coeff=1.0,
            img_shape=targets.shape[2:],
        )
        * config.kl_weight
    )

    net_loss = recons_loss + kl_loss  # TODO add check that losses coefs sum to 1
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


def microsplit_loss(
    model_outputs: tuple[torch.Tensor, dict[str, Any]],
    targets: torch.Tensor,
    config: LVAELossConfig,
    noise_model: "MultiChannelNoiseModel | None" = None,
    data_mean: float | None = None,
    data_std: float | None = None,
) -> dict[str, torch.Tensor] | None:
    """Unified loss function for MicroSplit (musplit, denoisplit, musplit_denoisplit).

    This function unifies the loss computation for all MicroSplit variants:
    - muSplit: gaussian_weight > 0, nm_weight = 0 (Gaussian likelihood only)
    - denoiSplit: nm_weight > 0, gaussian_weight = 0 (noise model likelihood only)
    - muSplit-denoiSplit: both weights > 0 (weighted combination)

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
        Uses `musplit_weight` as gaussian_weight and `denoisplit_weight` as nm_weight.
    noise_model : MultiChannelNoiseModel | None, optional
        The noise model. Required if denoisplit_weight > 0.
    data_mean : float | None, optional
        Data mean for denormalization. Required if denoisplit_weight > 0.
    data_std : float | None, optional
        Data std for denormalization. Required if denoisplit_weight > 0.

    Returns
    -------
    output : dict[str, torch.Tensor] | None
        A dictionary containing the overall loss `["loss"]`, the reconstruction loss
        `["reconstruction_loss"]`, and the KL divergence loss `["kl_loss"]`.
        Returns None if loss contains NaN values.
    """
    predictions, td_data = model_outputs
    img_shape = targets.shape[2:]

    gaussian_weight = config.musplit_weight
    nm_weight = config.denoisplit_weight

    if gaussian_weight > 0 and not config.predict_logvar:
        raise ValueError(
            "predict_logvar must be True in config when musplit_weight > 0"
        )
    if nm_weight > 0 and (noise_model is None or data_mean is None or data_std is None):
        raise ValueError(
            "noise_model, data_mean, and data_std required when denoisplit_weight > 0"
        )

    recons_loss: torch.Tensor | float = 0.0
    if nm_weight > 0 and gaussian_weight > 0:
        if predictions.shape[1] == 2 * targets.shape[1]:
            pred_mean, _ = predictions.chunk(2, dim=1)
        else:
            pred_mean = predictions

        recons_loss_nm = -_compute_noise_model_log_likelihood(
            reconstruction=pred_mean,
            target=targets,
            noise_model=noise_model,
            data_mean=data_mean,
            data_std=data_std,
        )
        recons_loss_gm = -_compute_gaussian_log_likelihood(
            reconstruction=predictions,
            target=targets,
            predict_logvar=config.predict_logvar,
            logvar_lowerbound=config.logvar_lowerbound,
        )
        recons_loss = nm_weight * recons_loss_nm + gaussian_weight * recons_loss_gm

    elif nm_weight > 0:
        recons_loss = -_compute_noise_model_log_likelihood(
            reconstruction=predictions,
            target=targets,
            noise_model=noise_model,
            data_mean=data_mean,
            data_std=data_std,
        )

    elif gaussian_weight > 0:
        recons_loss = -_compute_gaussian_log_likelihood(
            reconstruction=predictions,
            target=targets,
            predict_logvar=config.predict_logvar,
            logvar_lowerbound=config.logvar_lowerbound,
        )

    recons_loss = config.reconstruction_weight * recons_loss
    if isinstance(recons_loss, torch.Tensor) and torch.isnan(recons_loss).any():
        recons_loss = 0.0

    # Annealing is disabled, so kl_weight is just config.kl_weight
    if nm_weight > 0 and gaussian_weight > 0:
        # Combined mode: denoisplit uses kl_restricted, musplit uses kl
        denoisplit_kl = get_kl_divergence_loss(
            kl_type="kl_restricted",
            topdown_data=td_data,
            rescaling="image_dim",
            aggregation="sum",
            free_bits_coeff=1.0,
            img_shape=img_shape,
        )
        musplit_kl = get_kl_divergence_loss(
            kl_type="kl",
            topdown_data=td_data,
            rescaling="latent_dim",
            aggregation="mean",
            free_bits_coeff=0.0,
            img_shape=img_shape,
        )
        kl_loss = (nm_weight * denoisplit_kl + gaussian_weight * musplit_kl) * config.kl_weight

    elif nm_weight > 0:
        # Pure denoisplit mode
        kl_loss = (
            get_kl_divergence_loss(
                kl_type="kl_restricted",
                topdown_data=td_data,
                rescaling="image_dim",
                aggregation="sum",
                free_bits_coeff=1.0,
                img_shape=img_shape,
            )
            * config.kl_weight
        )

    else:
        # Pure musplit mode
        kl_loss = (
            get_kl_divergence_loss(
                kl_type="kl",
                topdown_data=td_data,
                rescaling="latent_dim",
                aggregation="mean",
                free_bits_coeff=0.0,
                img_shape=img_shape,
            )
            * config.kl_weight
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

    if torch.isnan(net_loss).any():
        return None

    return output
