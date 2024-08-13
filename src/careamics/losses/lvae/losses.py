##### REQUIRED Methods for Loss Computation #####
from typing import Optional, Union, Any

import numpy as np
import torch

from careamics.losses.lvae.loss_utils import free_bits_kl, get_kl_weight
from careamics.models.lvae.likelihoods import (
    LikelihoodModule,
    GaussianLikelihood,
    NoiseModelLikelihood
)
from careamics.losses.loss_factory import LVAELossParameters #TODO: may cause circular import
from careamics.models.lvae.utils import compute_batch_mean

Likelihood = Union[LikelihoodModule, GaussianLikelihood, NoiseModelLikelihood]

def get_reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    likelihood_obj: Likelihood,
    splitting_mask: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Compute the reconstruction loss.

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
    splitting_mask: Optional[torch.Tensor] = None
        A boolean tensor that indicates which items to keep for reconstruction loss
        computation. If `None`, all the elements of the items are considered
        (i.e., the mask is all `True`). Default is `None`.
    
    Returns
    -------
    dict[str, torch.Tensor]
        A dictionary containing the overall loss `["loss"]` and the loss for 
        individual output channels `["ch{i}_loss"]`.
    """
    loss_dict = _get_reconstruction_loss_vector(
        reconstruction=reconstruction,
        target=target,
        likelihood_obj=likelihood_obj,
    )

    if splitting_mask is None:
        splitting_mask = torch.ones_like(loss_dict["loss"]).bool()

    loss_dict["loss"] = loss_dict["loss"][splitting_mask].sum() / len(reconstruction)
    for i in range(1, 1 + target.shape[1]):
        key = f"ch{i}_loss"
        loss_dict[key] = loss_dict[key][splitting_mask].sum() / len(reconstruction)

    return loss_dict


def _get_reconstruction_loss_vector(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    likelihood_obj: LikelihoodModule,
) -> dict[str, torch.Tensor]:
    """Compute the reconstruction loss.

    Parameters
    ----------
    return_predicted_img: bool
        If set to `True`, the besides the loss, the reconstructed image is returned.
        Default is `False`.
    
    Returns
    -------
    dict[str, torch.Tensor]
        A dictionary containing the overall loss `["loss"]` and the loss for 
        individual output channels `["ch{i}_loss"]`. Shape of individual 
        tensors is (B, ).
    """                                                                                                     
    output = {"loss": None}
    for i in range(1, 1 + target.shape[1]):
        output[f"ch{i}_loss"] = None

    # Compute Log likelihood
    ll, _ = likelihood_obj(reconstruction, target) # shape: (B, C, [Z], Y, X)
    ll = _get_weighted_likelihood(ll) # TODO: needed?

    output = {"loss": compute_batch_mean(-1 * ll)} # shape: (B, )
    if ll.shape[1] > 1: # target_ch > 1
        for i in range(1, 1 + target.shape[1]):
            output[f"ch{i}_loss"] = compute_batch_mean(-ll[:, i - 1]) # shape: (B, )
    else: # target_ch == 1
        # TODO: hacky!!! Refactor this
        assert ll.shape[1] == 1
        output["ch1_loss"] = output["loss"]
        output["ch2_loss"] = output["loss"]

    return output


# TODO: check if this is correct
def reconstruction_loss_musplit_denoisplit(
    predictions,
    targets,
    predict_logvar,
    likelihood_NM,
    likelihood_GM,
    denoise_weight,
    split_weight,
):
    """.

    Parameters
    ----------
    predictions : _type_
        _description_
    targets : _type_
        _description_
    predict_logvar : _type_
        _description_
    likelihood_NM : _type_
        _description_
    likelihood_GM : _type_
        _description_
    denoise_weight : _type_
        _description_
    split_weight : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    if predict_logvar is not None:
        out_mean, _ = predictions.chunk(2, dim=1)
    else:
        out_mean = predictions

    recons_loss_nm = -1 * likelihood_NM(out_mean, targets)[0].mean()
    recons_loss_gm = -1 * likelihood_GM(predictions, targets)[0].mean()
    recons_loss = denoise_weight * recons_loss_nm + split_weight * recons_loss_gm
    return recons_loss

# TODO: refactor this (if needed)
# - cannot handle >2 target channels
# - cannot handle 3D inputs
def _get_weighted_likelihood(
    ll: torch.Tensor,
    ch1_recons_w: float = 1, 
    ch2_recons_w: float = 1
) -> torch.Tensor:
    """Multiply each channels with a different weight to get a weighted loss.
    
    Parameters
    ----------
    ll : torch.Tensor
        The log-likelihood tensor. Shape is (B, C, [Z], Y, X), where C is the number
        of channels.
    ch1_recons_w : float
        The weight for the first channel. Default is 1.
    ch2_recons_w : float
        The weight for the second channel. Default is 1.
    
    Returns
    -------
    torch.Tensor
        The weighted log-likelihood tensor. Shape is (B, C, [Z], Y, X).
    """
    if ch1_recons_w == 1 and ch2_recons_w == 1:
        return ll
    
    assert ll.shape[1] == 2, "This function is only for 2 channel images"

    mask1 = torch.zeros((len(ll), ll.shape[1], 1, 1), device=ll.device)
    mask1[:, 0] = 1
    mask2 = torch.zeros((len(ll), ll.shape[1], 1, 1), device=ll.device)
    mask2[:, 1] = 1

    return ll * mask1 * ch1_recons_w + ll * mask2 * ch2_recons_w


def get_kl_divergence_loss_usplit(
    topdown_layer_data_dict: dict[str, list[torch.Tensor]]
) -> torch.Tensor:
    """Compute the KL divergence loss for uSplit.
    
    Parameters
    ----------
    topdown_layer_data_dict : dict[str, list[torch.Tensor]]
        The top-down layer data dictionary containing the KL-loss values for each
        layer. The dictionary should contain the following keys:
        - "kl": The KL-loss values for each layer. Shape of each tensor is (B,).
        - "z": The sampled latents for each layer. Shape of each tensor is 
        (B, layers, `z_dims[i]`, H, W).
    """
    kl = torch.cat(
        [kl_layer.unsqueeze(1) for kl_layer in topdown_layer_data_dict["kl"]], dim=1
    ) # shape: (B, n_layers)
    # NOTE: Values are sum() and so are of the order 30000

    nlayers = kl.shape[1]
    for i in range(nlayers):
        # NOTE: we want to normalize the KL-loss w.r.t. the latent space dimensions,
        # i.e., the number of entries in the latent space tensors (C, [Z], Y, X).
        # We assume z has shape (B, C, [Z], Y, X), where `C = z_dims[i]`.
        norm_factor = np.prod(topdown_layer_data_dict["z"][i].shape[1:])
        kl[:, i] = kl[:, i] / norm_factor

    kl_loss = free_bits_kl(kl, 0.0).mean() # shape: (1, )
    return kl_loss


def get_kl_divergence_loss(topdown_layer_data_dict, img_shape, kl_key="kl"):
    # TODO explain all params!
    """kl[i] for each i has length batch_size resulting kl shape: (bs, layers)."""
    kl = torch.cat(
        [kl_layer.unsqueeze(1) for kl_layer in topdown_layer_data_dict[kl_key]],
        dim=1,
    )

    # As compared to uSplit kl divergence,
    # more by a factor of 4 just because we do sum and not mean.
    kl_loss = free_bits_kl(kl, 1).sum()  # TODO hardcoded free_bits 1
    # NOTE: at each hierarchy, it is more by a factor of 128/i**2).
    # 128/(2*2) = 32 (bottommost layer)
    # 128/(4*4) = 8
    # 128/(8*8) = 2
    # 128/(16*16) = 0.5 (topmost layer)

    # Normalize the KL-loss w.r.t. the  latent space
    kl_loss = kl_loss / np.prod(img_shape)
    return kl_loss


# Loss Computations
# mask = torch.isnan(target.reshape(len(x), -1)).all(dim=1)


def musplit_loss(
    model_outputs: tuple[torch.Tensor, dict[str, Any]], 
    targets: torch.Tensor, 
    loss_parameters: LVAELossParameters
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
    loss_parameters : LVAELossParameters
        The loss parameters for muSplit (e.g., KL hyperparameters, likelihood module,
        noise model, etc.).

    Returns
    -------
    output : Optional[dict[str, torch.Tensor]]
        A dictionary containing the overall loss `["loss"]`, the reconstruction loss
        `["reconstruction_loss"]`, and the KL divergence loss `["kl_loss"]`.
    """
    predictions, td_data = model_outputs
    recons_loss_dict = get_reconstruction_loss(
        reconstruction=predictions,
        target=targets,
        splitting_mask=loss_parameters.mask,
        likelihood_obj=loss_parameters.likelihood,
    )

    recons_loss = recons_loss_dict["loss"] * loss_parameters.reconstruction_weight

    if torch.isnan(recons_loss).any():
        recons_loss = 0.0

    kl_weight = get_kl_weight(
        loss_parameters.kl_annealing,
        loss_parameters.kl_start,
        loss_parameters.kl_annealtime,
        loss_parameters.kl_weight,
        loss_parameters.current_epoch,
    )
    kl_loss = kl_weight * get_kl_divergence_loss_usplit(td_data)

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


def denoisplit_loss(model_outputs, targets, loss_parameters) -> dict:
    """Loss function for DenoiSplit.

    Parameters
    ----------
    module : "CAREamicsModuleWrapper"
        "CAREamicsModuleWrapper" instance. This function is called from the
        "CAREamicsModuleWrapper".

    Returns
    -------
    dict
        _description_
    """
    # TODO what are all those params ? Document

    predictions, td_data = model_outputs
    recons_loss_dict, imgs = get_reconstruction_loss(
        reconstruction=predictions,
        target=targets,
        input=loss_parameters.inputs,
        splitting_mask=loss_parameters.mask,  # TODO splitting_mask is not used
        return_predicted_img=True,
        likelihood_obj=loss_parameters.likelihood,
    )

    recons_loss = recons_loss_dict["loss"] * loss_parameters.reconstruction_weight

    if torch.isnan(recons_loss).any():
        recons_loss = 0.0

    if loss_parameters.non_stochastic:  # TODO always false ?
        kl_loss = torch.Tensor([0.0]).cuda()
        net_loss = recons_loss
    else:
        # NOTE: 'kl' key stands for the 'kl_samplewise' key in the TopDownLayer class.
        # The different naming comes from `top_down_pass()` method in the LadderVAE.
        denoisplit_kl = get_kl_divergence_loss(
            topdown_layer_data_dict=td_data,
            img_shape=loss_parameters.inputs.shape,
            kl_key="kl",  # TODO hardcoded
        )
        musplit_kl = get_kl_divergence_loss_usplit(topdown_layer_data_dict=td_data)
        kl_loss = (
            loss_parameters.denoisplit_weight * denoisplit_kl
            + loss_parameters.musplit_weight * musplit_kl
        )
        # TODO self.kl_weight is hardcoded
        kl_loss = 1 * kl_loss

        recons_loss = reconstruction_loss_musplit_denoisplit(
            predictions,
            targets,
            predict_logvar=None,
            likelihood_NM=loss_parameters.noise_model,  # TODO is this correct ?
            likelihood_GM=loss_parameters.likelihood,
            denoise_weight=loss_parameters.denoisplit_weight,
            split_weight=loss_parameters.musplit_weight,
        )
        # recons_loss = _denoisplit_w * recons_loss_nm + _usplit_w * recons_loss_gm
        kl_loss = get_kl_weight(
            loss_parameters.kl_annealing,
            loss_parameters.kl_start,
            loss_parameters.kl_annealtime,
            loss_parameters.kl_weight,
            loss_parameters.current_epoch,
        ) * get_kl_divergence_loss(td_data, img_shape=loss_parameters.inputs.shape)

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
