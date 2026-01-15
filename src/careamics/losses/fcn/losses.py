"""
Loss submodule.

This submodule contains the various losses used in CAREamics.
"""

import torch
from torch.nn import L1Loss, MSELoss

from careamics.models.lvae.noise_models import GaussianMixtureNoiseModel


def mse_loss(source: torch.Tensor, target: torch.Tensor, *args) -> torch.Tensor:
    """
    Mean squared error loss.

    Parameters
    ----------
    source : torch.Tensor
        Source patches.
    target : torch.Tensor
        Target patches.
    *args : Any
        Additional arguments.

    Returns
    -------
    torch.Tensor
        Loss value.
    """
    loss = MSELoss()
    return loss(source, target)


def n2v_loss(
    manipulated_batch: torch.Tensor,
    original_batch: torch.Tensor,
    masks: torch.Tensor,
    *args,
) -> torch.Tensor:
    """
    N2V Loss function described in A Krull et al 2018.

    Parameters
    ----------
    manipulated_batch : torch.Tensor
        Batch after manipulation function applied. Shape: (B, C_out, ...)
    original_batch : torch.Tensor
        Original images. Shape: (B, C_in, ...)
    masks : torch.Tensor
        Coordinates of changed pixels. Shape: (B, C_in, ...)
    *args : Any
        Additional arguments.

    Returns
    -------
    torch.Tensor
        Loss value.

    Notes
    -----
    When C_out < C_in (e.g., model outputs 1 channel but input has multiple channels),
    the loss is computed only on channels where masks are non-zero (data channels).
    The manipulated_batch is broadcast or only the relevant channels are used.
    """
    # If output channels < input channels, only compute loss on masked channels
    if manipulated_batch.shape[1] < original_batch.shape[1]:
        # Find which channels have non-zero masks
        # Sum over all dimensions except channel dimension
        channel_has_mask = (
            masks.sum(dim=[d for d in range(len(masks.shape)) if d != 1]) > 0
        )

        # Get indices of channels with masks
        masked_channel_indices = torch.where(channel_has_mask)[0]

        # Select only the masked channels from original and masks
        original_batch = original_batch[:, masked_channel_indices, ...]
        masks = masks[:, masked_channel_indices, ...]

        # If model outputs 1 channel and there are multiple masked channels,
        # we need to expand the prediction to match
        if manipulated_batch.shape[1] == 1 and original_batch.shape[1] > 1:
            manipulated_batch = manipulated_batch.expand(
                -1, original_batch.shape[1], *[-1] * (len(original_batch.shape) - 2)
            )

    errors = (original_batch - manipulated_batch) ** 2
    # Average over pixels and batch
    loss = torch.sum(errors * masks) / torch.sum(masks)
    return loss  # TODO change output to dict ?


def pn2v_loss(
    samples: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    noise_model: GaussianMixtureNoiseModel,
) -> torch.Tensor:
    """
    Probabilistic N2V loss function described in A Krull et al., CVF (2019).

    Parameters
    ----------
    samples : torch.Tensor # TODO this naming is confusing
        Predicted pixel values from the network.
    labels : torch.Tensor
        Original pixel values.
    masks : torch.Tensor
        Coordinates of manipulated pixels.
    noise_model : GaussianMixtureNoiseModel
        Noise model for computing likelihood.

    Returns
    -------
    torch.Tensor
        Loss value.
    """
    likelihoods = noise_model.likelihood(labels, samples)
    likelihoods_avg = torch.log(torch.mean(likelihoods, dim=1, keepdim=True))

    # Average over pixels and batch
    loss = -torch.sum(likelihoods_avg * masks) / torch.sum(masks)
    return loss


def mae_loss(samples: torch.Tensor, labels: torch.Tensor, *args) -> torch.Tensor:
    """
    N2N Loss function described in to J Lehtinen et al 2018.

    Parameters
    ----------
    samples : torch.Tensor
        Raw patches.
    labels : torch.Tensor
        Different subset of noisy patches.
    *args : Any
        Additional arguments.

    Returns
    -------
    torch.Tensor
        Loss value.
    """
    loss = L1Loss()
    return loss(samples, labels)


# def dice_loss(
#     samples: torch.Tensor, labels: torch.Tensor, mode: str = "multiclass"
# ) -> torch.Tensor:
#     """Dice loss function."""
#     return DiceLoss(mode=mode)(samples, labels.long())
