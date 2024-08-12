##### REQUIRED Methods for Loss Computation #####
from typing import Optional

import numpy as np
import torch

from careamics.losses.lvae.loss_utils import free_bits_kl, get_kl_weight
from careamics.models.lvae.likelihoods import LikelihoodModule
from careamics.models.lvae.utils import compute_batch_mean


def get_reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    likelihood_obj: LikelihoodModule,
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
    splitting_mask: Optional[torch.Tensor] = None
        A boolean tensor that indicates which items to keep for reconstruction loss
        computation. If `None`, all the elements of the items are considered
        (i.e., the mask is all `True`). Default is `None`.
    likelihood_obj: LikelihoodModule = None
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
):
    """Compute the reconstruction loss.

    Parameters
    ----------
    return_predicted_img: bool
        If set to `True`, the besides the loss, the reconstructed image is returned.
        Default is `False`.
    """
    output = {"loss": None}
    for i in range(1, 1 + target.shape[1]):
        output[f"ch{i}_loss"] = None

    # Compute Log likelihood
    ll, _ = likelihood_obj(reconstruction, target)
    ll = _get_weighted_likelihood(ll)

    output = {"loss": compute_batch_mean(-1 * ll)}
    if ll.shape[1] > 1:
        for i in range(1, 1 + target.shape[1]):
            output[f"ch{i}_loss"] = compute_batch_mean(-ll[:, i - 1])
    else:
        assert ll.shape[1] == 1
        output["ch1_loss"] = output["loss"]
        output["ch2_loss"] = output["loss"]

    return output


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


def _get_weighted_likelihood(
    ll, ch1_recons_w=1, ch2_recons_w=1
):  # TODO what's this ? added defaults Thsi can be removed
    """Each of the channels gets multiplied with a different weight."""
    if ch1_recons_w == 1 and ch2_recons_w == 1:
        return ll
    # TODO Should these weights be removed?
    assert ll.shape[1] == 2, "This function is only for 2 channel images"

    mask1 = torch.zeros((len(ll), ll.shape[1], 1, 1), device=ll.device)
    mask1[:, 0] = 1
    mask2 = torch.zeros((len(ll), ll.shape[1], 1, 1), device=ll.device)
    mask2[:, 1] = 1

    return ll * mask1 * ch1_recons_w + ll * mask2 * ch2_recons_w


def get_kl_divergence_loss_usplit(
    topdown_layer_data_dict: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Function to compute the KL divergence loss for uSplit."""
    kl = torch.cat(
        [kl_layer.unsqueeze(1) for kl_layer in topdown_layer_data_dict["kl"]], dim=1
    )
    # NOTE: kl.shape = (16,4) 16 is batch size. 4 is number of layers.
    # Values are sum() and so are of the order 30000
    # Example values: 30626.6758, 31028.8145, 29509.8809, 29945.4922, 28919.1875

    nlayers = kl.shape[1]
    for i in range(nlayers):
        # topdown_layer_data_dict['z'][2].shape[-3:] = 128 * 32 * 32
        norm_factor = np.prod(topdown_layer_data_dict["z"][i].shape[-3:])
        # if _restricted_kl:
        #     pow = np.power(2,min(i + 1, _multiscale_count-1))
        #     norm_factor /= pow * pow

        kl[:, i] = kl[:, i] / norm_factor

    kl_loss = free_bits_kl(kl, 0.0).mean()
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


def musplit_loss(model_outputs, targets, loss_parameters) -> dict:
    """Loss function for MuSplit.

    Parameters
    ----------
    module : "CAREamicsModuleWrapper"
        This function is called from the "CAREamicsModuleWrapper".

    Returns
    -------
    dict
        _description_
    """
    predictions, td_data = model_outputs
    recons_loss_dict, imgs = get_reconstruction_loss(
        reconstruction=predictions,
        target=targets,
        input=loss_parameters.inputs,
        splitting_mask=loss_parameters.mask,
        return_predicted_img=True,
        likelihood_obj=loss_parameters.likelihood,
    )

    recons_loss = recons_loss_dict["loss"] * loss_parameters.reconstruction_weight

    if torch.isnan(recons_loss).any():
        recons_loss = 0.0

    kl_loss = get_kl_weight(
        loss_parameters.kl_annealing,
        loss_parameters.kl_start,
        loss_parameters.kl_annealtime,
        loss_parameters.kl_weight,
        loss_parameters.current_epoch,
    ) * get_kl_divergence_loss_usplit(td_data)

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
