"""Noise2Void and related losses."""

import torch

from careamics.models.lvae.noise_models import GaussianMixtureNoiseModel


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
        Batch after manipulation function applied.
    original_batch : torch.Tensor
        Original images.
    masks : torch.Tensor
        Coordinates of changed pixels.
    *args : Any
        Additional arguments.

    Returns
    -------
    torch.Tensor
        Loss value.
    """
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
