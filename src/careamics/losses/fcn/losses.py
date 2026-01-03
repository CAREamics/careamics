"""
Loss submodule.

This submodule contains the various losses used in CAREamics.
"""

import torch
import torch.nn.functional as F
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


def n2v_poisson_loss(
    manipulated_batch: torch.Tensor,  # model predictions (rates)
    original_batch: torch.Tensor,  # observed counts
    masks: torch.Tensor,
    *args,
) -> torch.Tensor:
    """
    N2V Loss with Poisson NLL for photon counting data.

    Uses PyTorch's optimized poisson_nll_loss with masked averaging.
    This implementation:
    - Leverages PyTorch's C++ backend for efficiency
    - Uses the same masking pattern as standard N2V loss
    - Computes mean loss over masked pixels only

    Parameters
    ----------
    manipulated_batch : torch.Tensor
        Predicted photon rates (must be positive). Shape: (B, C_out, ...)
    original_batch : torch.Tensor
        Observed photon counts. Shape: (B, C_in, ...)
    masks : torch.Tensor
        Binary mask indicating which pixels were masked. Shape: (B, C_in, ...)
    *args : Any
        Additional arguments.

    Returns
    -------
    torch.Tensor
        Scalar loss value (mean Poisson NLL over masked pixels).

    Notes
    -----
    The Poisson NLL formula is: λ - target*log(λ) (simplified, no factorial term).
    We compute this per-pixel, then take the mean over masked pixels only.

    Why sum(loss * mask) / sum(mask) instead of mean()?
    - Only ~2% of pixels are masked in N2V
    - We want the mean over MASKED pixels only, not all pixels
    - sum(loss * mask) / sum(mask) = weighted mean over masked pixels
    """
    # Handle channel dimension mismatch
    if manipulated_batch.shape[1] < original_batch.shape[1]:
        channel_has_mask = (
            masks.sum(dim=[d for d in range(len(masks.shape)) if d != 1]) > 0
        )
        masked_channel_indices = torch.where(channel_has_mask)[0]
        original_batch = original_batch[:, masked_channel_indices, ...]
        masks = masks[:, masked_channel_indices, ...]

        if manipulated_batch.shape[1] == 1 and original_batch.shape[1] > 1:
            manipulated_batch = manipulated_batch.expand(
                -1, original_batch.shape[1], *[-1] * (len(original_batch.shape) - 2)
            )

    # Check for empty masks early
    mask_sum = masks.sum()
    if mask_sum == 0:
        return torch.tensor(0.0, device=manipulated_batch.device, requires_grad=True)

    # Denormalize to photon count scale before Poisson NLL
    # Poisson requires non-negative counts, but normalized data can be negative!
    # Extract normalization stats from args if provided
    image_means = None
    image_stds = None
    if len(args) >= 2:
        image_means = args[0]
        image_stds = args[1]

    if image_means is not None and image_stds is not None:
        # Denormalize both predictions and targets to count scale
        device = manipulated_batch.device
        means = torch.tensor(image_means, device=device, dtype=manipulated_batch.dtype)
        stds = torch.tensor(image_stds, device=device, dtype=manipulated_batch.dtype)

        # Reshape for broadcasting: (1, C, 1, 1, ...) to match (B, C, D, H, W, ...)
        stats_shape = [1, len(means)] + [1] * (len(manipulated_batch.shape) - 2)
        means = means.view(stats_shape)
        stds = stds.view(stats_shape)

        # Denormalize: x_original = x_normalized * std + mean
        # Model outputs normalized values, we denormalize to photon count scale
        pred_denormalized = (manipulated_batch * stds) + means
        target_counts = (original_batch * stds) + means

        # Apply ReLU + epsilon to ensure predictions are positive (required for Poisson λ > 0)
        # ReLU has NO floor (unlike Softplus floor ~0.693), crucial for sparse data!
        # Small epsilon (1e-6) prevents log(0) in Poisson NLL computation
        pred_counts = F.relu(pred_denormalized) + 1e-6

        # Clamp targets to ensure non-negative (Poisson requirement)
        target_counts = target_counts.clamp(min=0.0)
    else:
        # No normalization - apply ReLU + epsilon (backward compatible)
        pred_counts = F.relu(manipulated_batch) + 1e-6
        target_counts = original_batch.clamp(min=0.0)

    # Compute Poisson NLL on denormalized photon count scale
    # Predictions are positive photon rates (after ReLU + epsilon)
    # Targets are clamped photon counts
    # This matches MSE's physical interpretation (both work in photon count space)
    nll_per_pixel = F.poisson_nll_loss(
        pred_counts,
        target_counts,
        log_input=False,  # Predictions are photon rates (not log rates)
        full=False,
        reduction='none'
    )

    # Apply mask and compute mean over masked pixels only
    loss = torch.sum(nll_per_pixel * masks) / mask_sum

    return loss


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
